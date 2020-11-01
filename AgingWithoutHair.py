import cv2
import numpy as np
import dlib
import PySimpleGUI as sg


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


oldLocation = "images/fAged.jpg"
myFaceLocation = "images/Jolie.jpg"

# GUI
sg.theme('Light Blue 2')

layout = [[sg.Text('Inserisci le immagini')],
          [sg.Text('OldFace', size=(8, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Text('YourFace', size=(8, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Submit(), sg.Cancel()]]
window = sg.Window('File Compare', layout)

event, values = window.read()
window.close()

if (values[0] != '') or (values[1] != '') :
    oldLocation = values[0]
    myFaceLocation = values[1]

oldFace = cv2.imread(oldLocation)
oldFaceGrayScale = cv2.cvtColor(oldFace, cv2.COLOR_BGR2GRAY)
maskOldFace = np.zeros_like(oldFaceGrayScale)

myFace = cv2.imread(myFaceLocation)
myFaceGrayScale = cv2.cvtColor(myFace, cv2.COLOR_BGR2GRAY)
maskMyFace = np.zeros_like(myFaceGrayScale)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")
height, width, channels = myFace.shape
oldFace_shapes_myFace = np.zeros((height, width, channels), np.uint8)

# Face old
faces = detector(oldFaceGrayScale)
for face in faces:
    landmarks = predictor(oldFaceGrayScale, face)
    landmarks_points_oldface = []
    for n in range(0, 81):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points_oldface.append((x, y))

    points = np.array(landmarks_points_oldface, np.int32)
    convexhullOldFace = cv2.convexHull(points)
    # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
    cv2.fillConvexPoly(maskOldFace, convexhullOldFace, 255)

    onlyOldFace = cv2.bitwise_and(oldFace, oldFace, mask=maskOldFace)

    # Delaunay triangulation
    rect = cv2.boundingRect(convexhullOldFace)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points_oldface)
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

# my Face
faces2 = detector(myFaceGrayScale)
for face in faces2:
    landmarks = predictor(myFaceGrayScale, face)
    landmarks_points_myface = []
    for n in range(0, 81):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points_myface.append((x, y))

    points2 = np.array(landmarks_points_myface, np.int32)
    convexhullMyFace = cv2.convexHull(points2)

    cv2.fillConvexPoly(maskMyFace, convexhullMyFace, 255)
    onlyMyFace = cv2.bitwise_and(myFace, myFace, mask=maskMyFace)

lines_space_mask = np.zeros_like(oldFaceGrayScale)
lines_space_new_face = np.zeros_like(myFace)

# Triangulation of both faces
for triangle_index in indexes_triangles:
    # Triangulation of the old face
    tr1_pt1 = landmarks_points_oldface[triangle_index[0]]
    tr1_pt2 = landmarks_points_oldface[triangle_index[1]]
    tr1_pt3 = landmarks_points_oldface[triangle_index[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

    rect1 = cv2.boundingRect(triangle1)
    (x, y, w, h) = rect1
    cropped_triangle = oldFace[y: y + h, x: x + w]
    cropped_tr1_mask = np.zeros((h, w), np.uint8)

    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

    # Lines space
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
    cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
    lines_space = cv2.bitwise_and(oldFace, oldFace, mask=lines_space_mask)

    # Triangulation of my face
    tr2_pt1 = landmarks_points_myface[triangle_index[0]]
    tr2_pt2 = landmarks_points_myface[triangle_index[1]]
    tr2_pt3 = landmarks_points_myface[triangle_index[2]]
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
    img2_new_face_rect_area = oldFace_shapes_myFace[y: y + h, x: x + w]
    img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

    img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
    oldFace_shapes_myFace[y: y + h, x: x + w] = img2_new_face_rect_area

# lascia invariati gli occhi
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

mixFaces = cv2.addWeighted(onlyMyFace, 0.4, oldFace_shapes_myFace, 0.6, 0)

result = cv2.add(myFace_withoutFace, mixFaces)
(x, y, w, h) = cv2.boundingRect(convexhullMyFace)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

myFaceAged = cv2.seamlessClone(result, myFace, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
cv2.imshow("Aged", myFaceAged)

cv2.waitKey(0)

cv2.destroyAllWindows()
