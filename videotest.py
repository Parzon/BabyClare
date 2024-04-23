import cv2
import os

def capture_frames_from_camera(frame_rate, save_path):
    # Ensure save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory at {save_path}")

    video_capture = cv2.VideoCapture(0)  # '0' for default camera
    count = 0  # Frame counter

    while True:
        success, frame = video_capture.read()
        if not success:
            print("Failed to capture video. Exiting...")
            break

        current_frame = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame % frame_rate == 0:
            frame_filename = f"{save_path}/frame_{count}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")  # Confirm each save
            count += 1

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting capture...")
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return count

if __name__ == '__main__':
    frame_rate = 30  # Modify as needed
    saved_frames = capture_frames_from_camera(frame_rate, './captured_frames')
    print(f"Total frames saved: {saved_frames}")


# interrupt the code with ctrl+c
