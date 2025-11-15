import cv2

drawing = False
ix, iy = -1, -1
rectangles = []
img = None
scale_x = scale_y = 1.0


def resize_for_display(image, max_width=1000, max_height=700):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    return cv2.resize(image, (int(w * scale), int(h * scale))), scale


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, rectangles, temp_img, scale_x, scale_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        temp_img = img.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = temp_img.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

        # Scale rectangle back to original image coordinates
        rect = (
            int(ix / scale_x),
            int(iy / scale_y),
            int(x / scale_x),
            int(y / scale_y),
        )
        rectangles.append(rect)

        print(f"Rectangle in original size: {rect}")


def main(image_path):
    global img, temp_img, scale_x, scale_y

    img_original = cv2.imread(image_path)
    if img_original is None:
        print("Error: Image not found!")
        return

    img, scale = resize_for_display(img_original)
    scale_x = scale_y = scale
    temp_img = img.copy()

    cv2.namedWindow("Draw Rectangles")
    cv2.setMouseCallback("Draw Rectangles", draw_rectangle)

    print("\nDraw rectangles with mouse.")
    print("Press 'q' to quit.\n")

    while True:
        cv2.imshow("Draw Rectangles", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    print("\nAll Rectangles in original resolution:")
    for i, r in enumerate(rectangles):
        print(f"{i+1}. {r}")


if __name__ == "__main__":
    main("./images/swan.jpg")
