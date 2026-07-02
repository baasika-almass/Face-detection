import cv2
import os

# Load OpenCV's pre-trained face detection model (comes built-in with OpenCV)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(image):
    """
    Detect faces in an image and draw rectangles around them.
    Returns the modified image and the number of faces found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,      # how much the image size is reduced at each scale
        minNeighbors=5,       # how many neighbors a rectangle needs to be kept
        minSize=(30, 30)      # minimum possible face size
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image, "Face", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    return image, len(faces)


def detect_from_photo():
    """Detect faces in a photo file."""
    path = input("\n  Enter the path to your photo (e.g. photo.jpg): ").strip()

    if not os.path.exists(path):
        print("  ❌ File not found! Make sure the path is correct.")
        return

    image = cv2.imread(path)

    if image is None:
        print("  ❌ Could not read the image. Make sure it's a valid image file.")
        return

    print("  🔍 Detecting faces...")
    result_image, count = detect_faces(image)

    print(f"  ✅ Found {count} face(s)!")

    output_path = "output_" + os.path.basename(path)
    cv2.imwrite(output_path, result_image)
    print(f"  💾 Saved result as '{output_path}'")

    cv2.imshow("Face Detection - Press any key to close", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_from_webcam():
    """Detect faces in real-time from the webcam."""
    print("\n  📷 Starting webcam...")
    print("  👉 Click on the video window first, then press 'q' to quit.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("  ❌ Could not access webcam. Check if it's connected and not in use.")
        return

    window_name = "Webcam Face Detection - Click here, then press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  ❌ Failed to grab frame from webcam.")
            break

        frame, count = detect_faces(frame)

        cv2.putText(
            frame, f"Faces detected: {count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
        )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC key
            break

        # Also stop if the window was closed using the X button
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  👋 Webcam closed.")


def show_menu():
    print("\n  📸 FACE DETECTION (OpenCV)")
    print("  " + "=" * 30)
    print("  1. Detect faces in a photo")
    print("  2. Detect faces from webcam")
    print("  3. Exit")


def main():
    print("\n  Welcome to Face Detection!")

    while True:
        show_menu()
        choice = input("\n  Choose an option (1-3): ").strip()

        if choice == "1":
            detect_from_photo()
        elif choice == "2":
            detect_from_webcam()
        elif choice == "3":
            print("\n  Goodbye! 👋\n")
            break
        else:
            print("  Invalid choice. Please enter 1-3.")


if __name__ == "__main__":
    main()
