from collections import deque

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


INPUT_STRIDE = 32
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NEIGHBORS_8 = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),             (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
)


class PPOCRv6Detector:
    def __init__(self, model_path, score_thresh=0.3, min_area=20):
        self.score_thresh = score_thresh
        self.min_area = min_area
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image):
        original_width, original_height = image.size
        input_width = ((original_width + INPUT_STRIDE - 1) // INPUT_STRIDE) * INPUT_STRIDE
        input_height = ((original_height + INPUT_STRIDE - 1) // INPUT_STRIDE) * INPUT_STRIDE

        resized = image.resize((input_width, input_height))
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        pixels = (pixels - RGB_MEAN) / RGB_STD
        tensor = np.transpose(pixels, (2, 0, 1))[None, ...]

        return tensor, (original_width, original_height)

    def infer(self, image):
        tensor, original_size = self.preprocess(image)
        output = self.session.run(None, {self.input_name: tensor})[0]
        return output[0, 0], original_size

    def extract_boxes(self, score_map, original_size):
        mask = score_map > self.score_thresh
        map_height, map_width = mask.shape
        original_width, original_height = original_size
        scale_x = original_width / map_width
        scale_y = original_height / map_height
        visited = np.zeros(mask.shape, dtype=bool)
        boxes = []

        for start_y in range(map_height):
            for start_x in range(map_width):
                if not mask[start_y, start_x] or visited[start_y, start_x]:
                    continue

                pending = [(start_x, start_y)]
                visited[start_y, start_x] = True
                area = 0
                min_x = max_x = start_x
                min_y = max_y = start_y

                while pending:
                    x, y = pending.pop()
                    area += 1
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)

                    for offset_x, offset_y in NEIGHBORS_8:
                        neighbor_x = x + offset_x
                        neighbor_y = y + offset_y

                        if not (0 <= neighbor_x < map_width and 0 <= neighbor_y < map_height):
                            continue
                        if not mask[neighbor_y, neighbor_x] or visited[neighbor_y, neighbor_x]:
                            continue

                        visited[neighbor_y, neighbor_x] = True
                        pending.append((neighbor_x, neighbor_y))

                if area < self.min_area:
                    continue

                component_width = max_x - min_x + 1
                component_height = max_y - min_y + 1
                if component_width <= component_height:
                    continue

                x1 = int(min_x * scale_x)
                y1 = int(min_y * scale_y)
                x2 = int(max_x * scale_x)
                y2 = int(max_y * scale_y)
                horizontal_padding = 2
                vertical_padding = max(4, int((x2 - x1) * 0.025))

                boxes.append((
                    max(0, x1 - horizontal_padding),
                    max(0, y1 - vertical_padding),
                    min(original_width - 1, x2 + horizontal_padding),
                    min(original_height - 1, y2 + vertical_padding),
                ))

        return boxes

    def predict(self, image_path):
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        score_map, original_size = self.infer(image)
        boxes = self.extract_boxes(score_map, original_size)
        return image, boxes


def main():
    detector = PPOCRv6Detector("inference.onnx", score_thresh=0.30, min_area=30)
    image, boxes = detector.predict("test.png")
    draw = ImageDraw.Draw(image)

    for box in boxes:
        draw.rectangle(box, outline="red", width=2)
        print(box)

    image.save("result.png")
    print(f"Detected TextLines: {len(boxes)}")


if __name__ == "__main__":
    main()