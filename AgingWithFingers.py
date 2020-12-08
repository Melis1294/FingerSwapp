import cv2
import matplotlib.pyplot as plt
import dlib
import numpy as np
import math
import pygame
import time

def swappedFace(path):
    face = cv2.imread(path)
    faceGrey = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(faceGrey)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    faces = detector(faceGrey)
    landmarks = predictor(faceGrey, faces[0])
    landmarks_points_face = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points_face.append((x, y))
    points = np.array(landmarks_points_face, np.int32)
    convexHullFace = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexHullFace, 255)

    # Delaunay triangulation
    rect = cv2.boundingRect(convexHullFace)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points_face)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
    return indexes_triangles, landmarks_points_face, face


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

#play_sound fuction uses the pygame library
def play_sound(filename,wait): # play an existing wav format sound, wait stand for if you want the voice to be played and to wait executing any othe code or not
    filepath = filename # make sure to set where your file is located for example here home/Document p.s you dont need to write home
    pygame.mixer.init()
    pygame.mixer.music.load(filepath)
    pygame.mixer.music.play()
    if wait:
        while pygame.mixer.music.get_busy() == True:
            continue

def give_hud(frame, logo):
        start_roi_row = frame.shape[1]-logo.shape[1]
        start_roi_col = logo.shape[0]
        image_roi = frame[0:start_roi_col, start_roi_row:frame.shape[1]]

        result = cv2.add(image_roi, logo)
        frame[0:start_roi_col, start_roi_row:frame.shape[1]] = result
        return frame

"""def screenshot(TIMER, prev, myFace, filenumber, hud):
        while TIMER >= 0:
            # Display countdown on each frame
            # specify the font and draw the
            # countdown using puttext
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(myFace, str(TIMER),
                        (200, 250), font,
                        7, (0, 255, 255),
                        4, cv2.LINE_AA)
            cv2.imshow('a', myFace)
            cv2.waitKey(125)

            # current time
            cur = time.time()

            # Update and keep track of Countdown
            # if time elapsed is one second
            # than decrese the counter
            if cur-prev >= 1:
                prev = cur
                TIMER = TIMER-1
                hud = 0

            else:
                # Save the frame
                cv2.imwrite('shots/image%2d.jpeg' % filenumber, myFace)
                play_sound('audios/shutter.mp3',True) #True = riproduce tutta la traccia audio
                filenumber += 1
                TIMER = int(3)
                hud = 1

        return  TIMER, myFace, hud"""

def main():
    indexes_triangles_woman, landmarks_points_face_woman, face_woman = swappedFace("images/Woman.jpg")
    indexes_triangles_man, landmarks_points_face_man, face_man = swappedFace("images/Man.jpg")
    indexes_triangles_oldwoman, landmarks_points_face_oldwoman, face_oldwoman = swappedFace("images/OldWoman.jpg")
    indexes_triangles_oldman, landmarks_points_face_oldman, face_oldman = swappedFace("images/oldman.jpg")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    plt.ion()
    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))
    img = None

    flag = 0
    flag1= 0
    filenumber = 0
    hud = 1
    # SET THE COUNTDOWN TIMER
    TIMER = int(4)
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    first = 1

    while cap.isOpened():
        myFace = grab_frame(cap)
        print("TIMER: ", TIMER-1)

        if(hud == 1):
            screen = 0
            img_pre = cv2.imread('images/rotation_wheel.png')
            img_post = cv2.resize(img_pre, (150, 150))

            myFace = give_hud(myFace, img_post)

            # Hand Gesture handle
            # Definisco una regione di interesse per prendere informazioni della mano
            roi = myFace[10:200, 10:200]
            cv2.rectangle(myFace, (10, 10), (200, 200), (0, 255, 0), 0)
            hands_on = 1
        else:
            screen = 1
            hands_on = 0

        if(hands_on == 1):
            # Maschero solo la zona della mano
            redChannel = roi[:, :, 2]
            _, thresh_gray = cv2.threshold(redChannel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            maskHand = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            maskHand = cv2.morphologyEx(maskHand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            # Trovo il contorno della mano
            contours, hierarchy = cv2.findContours(maskHand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            try:
                # Prendo solo il conorno che ha l'area più grande
                cnt = max(contours, key=lambda x: cv2.contourArea(x))
                epsilon = 0.0005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                hull = cv2.convexHull(cnt)
                areahull = cv2.contourArea(hull)
                areacnt = cv2.contourArea(cnt)
                arearatio=((areahull-areacnt)/areacnt)*100
                hull = cv2.convexHull(approx, returnPoints=False)
                defects = cv2.convexityDefects(approx, hull)
                fingers = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])
                    pt = (100, 180)

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
                    indexes_triangles = indexes_triangles_woman
                    punti_landmakr_otherface = landmarks_points_face_woman
                    oldFace = face_woman
                    flag = 1
                    flag1 = 1

                elif fingers == 2:
                    indexes_triangles = indexes_triangles_man
                    punti_landmakr_otherface = landmarks_points_face_man
                    oldFace = face_man
                    flag = 1
                    flag1 = 1
                elif fingers == 3:
                    indexes_triangles = indexes_triangles_oldwoman
                    punti_landmakr_otherface = landmarks_points_face_oldwoman
                    oldFace = face_oldwoman
                    flag = 1
                    flag1 = 1
                elif fingers == 4:
                    indexes_triangles = indexes_triangles_oldman
                    punti_landmakr_otherface = landmarks_points_face_oldman
                    oldFace = face_oldman
                    flag = 1
                    flag1 = 1
                elif fingers == 5:
                    hud = 0     #disattivo l'hud prima di fare lo screen
            except Exception as e:
                pass

        # My Face handle
        if flag == 1:
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
                    tr1_pt1 = punti_landmakr_otherface[triangle_index[0]]
                    tr1_pt2 = punti_landmakr_otherface[triangle_index[1]]
                    tr1_pt3 = punti_landmakr_otherface[triangle_index[2]]
                    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                    rect1 = cv2.boundingRect(triangle1)
                    (x, y, w, h) = rect1
                    cropped_triangle = oldFace[y: y + h, x: x + w]
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
                    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
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

                # Face swapped (putting 1st face into 2nd face)
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
            if screen == 1:
                if(first == 1):
                    prev = time.time()
                    first = 0
                if(TIMER>0):
                    text = str(TIMER-1)
                    # get boundary of this text
                    textsize = cv2.getTextSize(text, font, 7, 2)[0]
                    # get coords based on boundary
                    textX = int((myFace.shape[1] - textsize[0]) / 2)
                    textY = int((myFace.shape[0] + textsize[1]) / 2)
                    cv2.putText(myFace, text, (textX, textY), font, 7, (0, 255, 255), 2)
                    play_sound('audios/beep.mp3', False)      #True = riproduce tutta la traccia audio
                # current time
                cur = time.time()
                if cur-prev >= 1:
                    prev = cur
                    TIMER = TIMER-1
                if TIMER < 0:
                    cv2.imwrite('shots/image%2d.jpeg' % filenumber, myFace)
                    play_sound('audios/shutter.mp3', True) #True = riproduce tutta la traccia audio
                    filenumber += 1
                    TIMER = int(4)
                    hud = 1
                    first = 1
                if flag1 == 1:
                    flag1 = 0   #cos'è? [KLA]

        if img is None:
            img = plt.imshow(cv2.cvtColor(myFace, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("Camera Capture")
            #cv2.waitKey(2000)
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
