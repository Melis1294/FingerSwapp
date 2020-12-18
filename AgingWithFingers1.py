import matplotlib.pyplot as plt
import math
import pygame
import time
from data import *


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def grab_frame(cap):
    """
    Method to grab a frame from the camera
    :param cap: the VideoCapture object
    :return: the captured image
    """
    ret, myFace = cap.read()
    myFace = cv2.flip(myFace, 1)
    return myFace


def handle_close(event, cap):
    """
    Handle the close event of the Matplotlib window by closing the camera capture
    :param event: the close event
    :param cap: the VideoCapture object to be closed
    """
    cap.release()


def give_hud(frame, frame_bg, logo_bg, start_roi_col, start_roi_row):
    result = cv2.add(frame_bg, logo_bg)
    frame[0:start_roi_col, start_roi_row:frame.shape[1]] = result
    return frame


def play_sound(filename,
               wait):  # play an existing wav format sound, wait stand for if you want the voice to be played and to wait executing any othe code or not
    filepath = filename  # make sure to set where your file is located for example here home/Document p.s you dont need to write home
    pygame.mixer.init()
    pygame.mixer.music.load(filepath)
    pygame.mixer.music.play()
    if wait:
        while pygame.mixer.music.get_busy() == True:
            continue


def main():
    cap = cv2.VideoCapture(0)
    plt.ion()
    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))
    img = None
    # Flag che serve per far partire il programma senza necessità che si abbiano le dita alzate.
    fingers_up = 0
    # Flag che serve a far sì che non vengano fatti continuamente screen quando si fa 5 con la mano
    # screen_fatto = 0
    # Contatore per salvare le immagini
    filenumber = 0

    # Serve per disegnare le cose nel fram
    draw_frame = 1
    TIMER = int(4)
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    first = 1

    # Serve per fare i calcoli del logo solo una volta e non sempre, evitando così di appesantire il programma
    frame = grab_frame(cap)
    img_pre = cv2.imread('images/rotation_wheel.png')
    logo = cv2.resize(img_pre, (100, 100))
    start_roi_row = frame.shape[1] - logo.shape[1]
    start_roi_col = logo.shape[0]
    image_roi = frame[0:start_roi_col, start_roi_row:frame.shape[1]]
    logogray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(logogray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(image_roi, image_roi, mask=mask_inv)
    logo_bg = cv2.bitwise_and(logo, logo, mask=mask)

    while cap.isOpened():
        myFace = grab_frame(cap)

        # Disegna le cose nel frame (logo, riquadro), non fa screen, è il caso in cui ho la mano alzata
        if draw_frame == 1:
            screen = 0
            myFace = give_hud(myFace, frame_bg, logo_bg, start_roi_col, start_roi_row)
            # Hand Gesture handle
            # Definisco una regione di interesse per prendere informazioni della mano
            roi = myFace[100:300, 100:300]
            cv2.rectangle(myFace, (100, 100), (300, 300), (0, 255, 0), 0)
        else: # Caso in cui non ho mani alzate, e devo fare uno screen.
            screen = 1

        # Se devo fare uno screen, non devo fare il calcolo delle mani, altrimenti sì.
        if screen == 0:
            # Maschero solo la zona della mano
            redChannel = roi[:, :, 2]
            _, thresh_gray = cv2.threshold(redChannel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            maskHand = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            maskHand = cv2.morphologyEx(maskHand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            # Trovo il contorno della mano
            contours, hierarchy = cv2.findContours(maskHand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            try:
                # Prendo solo il conorno che ha l'area più grande
                cnt = max(contours, key=lambda x: cv2.contourArea(x))
                epsilon = 0.0005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                hull = cv2.convexHull(approx, returnPoints=False)
                defects = cv2.convexityDefects(approx, hull)
                fingers = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])
                    # pt = (100, 180)

                    # find length of all sides of triangle
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    s = (a + b + c) / 2
                    ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                    # distance between point and convex hull
                    d = (2 * ar) / a

                    # apply cosine rule here
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                    # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                    if angle <= 90 and d > 30:
                        fingers += 1
                        cv2.circle(roi, far, 3, [255, 0, 0], -1)

                    # draw lines around hand
                    cv2.line(roi, start, end, [0, 255, 0], 2)
                fingers += 1

                if fingers == 1:
                    cv2.putText(myFace, '1 finger - Woman Face', (0, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    indexes_triangles = indexes_triangles_woman
                    punti_landmark_otherface = landmarks_points_face_woman
                    otherFace = face_woman
                    fingers_up = 1

                elif fingers == 2:
                    cv2.putText(myFace, '2 finger - Man Face', (0, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    indexes_triangles = indexes_triangles_man
                    punti_landmark_otherface = landmarks_points_face_man
                    otherFace = face_man
                    fingers_up = 1
                elif fingers == 3:
                    cv2.putText(myFace, '3 finger - Old woman face', (0, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    indexes_triangles = indexes_triangles_oldwoman
                    punti_landmark_otherface = landmarks_points_face_oldwoman
                    otherFace = face_oldwoman
                    fingers_up = 1
                elif fingers == 4:
                    cv2.putText(myFace, '4 finger - Old man face', (0, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    indexes_triangles = indexes_triangles_oldman
                    punti_landmark_otherface = landmarks_points_face_oldman
                    otherFace = face_oldman
                    fingers_up = 1
                elif fingers == 5:
                    # Cancella i disegni
                    draw_frame = 0
                else:
                    fingers_up = 0
            except Exception as e:
                pass

        # Serve per la prima volta: se non ho alzato nessun dito, non avrei i valori punti_landmark, indexes_triangles e oldface
        # Gestione della mia faccia
        if fingers_up == 1:
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            # Converto la mia faccia in scala di grigi
            myFaceGrayScale = cv2.cvtColor(myFace, cv2.COLOR_BGR2GRAY)
            # Creo una'immagine nera della stessa dimensione del frame
            maskMyFace = np.zeros_like(myFaceGrayScale)
            height, width, channels = myFace.shape
            # creo un'immagine con le stesse dimensioni e canali del mio frame
            otherFace_shapes_myface = np.zeros((height, width, channels), np.uint8)
            # Salvo nella variabile faces il numero di facce che trovo
            faces2 = detector(myFaceGrayScale)
            if len(faces2) > 0:
                for face in faces2:
                    # Salvo in questa variabile la posizion x,y di tutti i punti della mia faccia.
                    punti_landmark_myFace = []
                    landmarks = predictor(myFaceGrayScale, face)
                    for n in range(0, 68):
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        punti_landmark_myFace.append((x, y))
                    # Creo un convexhull a partire dai punti definiti precedentemente
                    points2 = np.array(punti_landmark_myFace, np.int32)
                    convexhullMyFace = cv2.convexHull(points2)
                    # Riempo il convexhull con il colore bianco nell'immagine nera con 1 solo canale
                    cv2.fillConvexPoly(maskMyFace, convexhullMyFace, 255)
                    # Variabile con solo la mia faccia
                    onlyMyFace = cv2.bitwise_and(myFace, myFace, mask=maskMyFace)

                for triangle_index in indexes_triangles:
                    # Triangulation of the other face
                    tr1_pt1 = punti_landmark_otherface[triangle_index[0]]
                    tr1_pt2 = punti_landmark_otherface[triangle_index[1]]
                    tr1_pt3 = punti_landmark_otherface[triangle_index[2]]
                    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                    rect1 = cv2.boundingRect(triangle1)
                    (x, y, w, h) = rect1
                    cropped_triangle = otherFace[y: y + h, x: x + w]
                    cropped_tr1_mask = np.zeros((h, w), np.uint8)

                    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

                    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

                    # Triangulation of my face
                    tr2_pt1 = punti_landmark_myFace[triangle_index[0]]
                    tr2_pt2 = punti_landmark_myFace[triangle_index[1]]
                    tr2_pt3 = punti_landmark_myFace[triangle_index[2]]
                    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

                    rect2 = cv2.boundingRect(triangle2)
                    (x, y, w, h) = rect2

                    cropped_tr2_mask = np.zeros((h, w), np.uint8)

                    points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                        [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

                    # Warp triangles
                    points = np.float32(points)
                    points2 = np.float32(points2)
                    M = cv2.getAffineTransform(points, points2)
                    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

                    # Reconstructing destination face
                    img2_new_face_rect_area = otherFace_shapes_myface[y: y + h, x: x + w]
                    img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
                    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255,
                                                               cv2.THRESH_BINARY_INV)
                    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

                    img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
                    otherFace_shapes_myface[y: y + h, x: x + w] = img2_new_face_rect_area

                left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                            (landmarks.part(37).x, landmarks.part(37).y),
                                            (landmarks.part(38).x, landmarks.part(38).y),
                                            (landmarks.part(39).x, landmarks.part(39).y),
                                            (landmarks.part(40).x, landmarks.part(40).y),
                                            (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

                right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                             (landmarks.part(43).x, landmarks.part(43).y),
                                             (landmarks.part(44).x, landmarks.part(44).y),
                                             (landmarks.part(45).x, landmarks.part(45).y),
                                             (landmarks.part(46).x, landmarks.part(46).y),
                                             (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

                teeth_region = np.array([(landmarks.part(60).x, landmarks.part(60).y),
                                         (landmarks.part(61).x, landmarks.part(61).y),
                                         (landmarks.part(62).x, landmarks.part(62).y),
                                         (landmarks.part(63).x, landmarks.part(63).y),
                                         (landmarks.part(64).x, landmarks.part(64).y),
                                         (landmarks.part(65).x, landmarks.part(65).y),
                                         (landmarks.part(66).x, landmarks.part(66).y),
                                         (landmarks.part(67).x, landmarks.part(67).y)])

                # Sovrapposizione delle facce
                img2_face_mask = np.zeros_like(myFaceGrayScale)
                img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhullMyFace, 255)
                cv2.fillPoly(img2_head_mask, [left_eye_region], 0)
                cv2.fillPoly(img2_head_mask, [right_eye_region], 0)
                cv2.fillPoly(img2_head_mask, [teeth_region], 0)
                img2_face_mask = cv2.bitwise_not(img2_head_mask)
                myFace_withoutFace = cv2.bitwise_and(myFace, myFace, mask=img2_face_mask)

                mixFaces = cv2.addWeighted(onlyMyFace, 0.4, otherFace_shapes_myface, 0.6, 0)

                result = cv2.add(myFace_withoutFace, mixFaces)
                (x, y, w, h) = cv2.boundingRect(convexhullMyFace)
                center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
                myFace = cv2.seamlessClone(result, myFace, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
            # Se devo fare lo screen
            if screen == 1:
                if first == 1:
                    prev = time.time()
                    first = 0
                if TIMER > 0:
                    text = str(TIMER - 1)
                    textsize = cv2.getTextSize(text, font, 7, 2)[0]
                    # get coords based on boundary
                    textX = int((myFace.shape[1] - textsize[0]) / 2)
                    textY = int((myFace.shape[0] + textsize[1]) / 2)
                    cv2.putText(myFace, text, (textX, textY), font, 7, (0, 255, 255), 2)
                    play_sound('audios/beep.mp3', False)  # True = riproduce tutta la traccia audio
                cur = time.time()
                if cur - prev >= 1:
                    prev = cur
                    TIMER = TIMER - 1
                if TIMER < 0:
                    cv2.imwrite('shots/image%2d.jpeg' % filenumber, myFace)
                    play_sound('audios/shutter.mp3', True)  # True = riproduce tutta la traccia audio
                    filenumber += 1
                    TIMER = int(4)
                    draw_frame = 1
                    first = 1

        if img is None:
            img = plt.imshow(cv2.cvtColor(myFace, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("Camera Capture")
            plt.show()
        else:
            img.set_data(cv2.cvtColor(myFace, cv2.COLOR_BGR2RGB))
            fig.canvas.draw()
            fig.canvas.flush_events()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
