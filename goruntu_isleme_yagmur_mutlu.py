
import cv2
import mediapipe as mp

### resimin el tespiti yapılmış görsel halinin hesaplanmasını içeren fonksiyonu yazın ###
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.multi_hand_landmarks
    handedness_list = detection_result.multi_handedness
    annotated_image = rgb_image.copy()

    if hand_landmarks_list is not None:
        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            handedness = handedness_list[idx].classification[0].label

            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
            y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            cv2.putText(
                annotated_image,
                f"{handedness}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA
            )

    return annotated_image


if __name__ == '__main__':
    # elinizi içeren resmi (el_tespiti_1.jpg) OpenCV ile yükleyin.
    image = cv2.imread("C:\Users\Lenovo\OneDrive\Masaüstü\data\el_tespiti_1.jpg")

    ### resim yüksekliğini 960, genişliğini 640 olarak düzenleyin. ###
    image_resized = cv2.resize(image, (960, 640))

    ### resim üzerine 10x10 blur uygulayın. ###
    image_blurred = cv2.blur(image_resized, (10, 10))

    ### resmi BGR formatından RGB formatına dönüştürün. ###
    image_rgb = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2RGB)

    ### resmi "el_tespit_odev.jpg" olarak kaydedin. ###
    cv2.imwrite("C:\Users\Lenovo\OneDrive\Masaüstü\data\el_tespit_odev.jpg", image_rgb)

    # el_tespit_odev.jpg resmini OpenCV ile okuyun.
    image = cv2.imread("C:\Users\Lenovo\OneDrive\Masaüstü\data\el_tespit_odev.jpg")

    MARGIN = 10  # metnin üst köşesinden elin sınırına olan mesafe
    FONT_SIZE = 3  # yazı tipi boyutu
    FONT_THICKNESS = 2  # yazı kalınlığı
    HANDEDNESS_TEXT_COLOR = (255, 0, 0)  # kırmızı renk

    hands = mp.solutions.hands.Hands(max_num_hands=2)

    ### resmin el tespiti yapılmış görsel halini oluşturun ###
    detection_result = hands.process(image)

    ### resmin el tespiti yapılmış görsel halini hesaplayın ###
    image_with_landmarks = draw_landmarks_on_image(image, detection_result)

    ### resmin el tespiti yapılmış görsel halini BGR formatına dönüştürün ###
    image_with_landmarks_bgr = cv2.cvtColor(image_with_landmarks, cv2.COLOR_RGB2BGR)

    # resmi "el_tespit_odev.jpg" olarak kaydedin.
    cv2.imwrite("C:\Users\Lenovo\OneDrive\Masaüstü\data\el_tespit_odev.jpg", image_with_landmarks_bgr)