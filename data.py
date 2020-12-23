import dlib
import cv2
import numpy as np

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

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

indexes_triangles_woman, landmarks_points_face_woman, face_woman = swappedFace("images/Woman.jpg")
indexes_triangles_man, landmarks_points_face_man, face_man = swappedFace("images/Man.jpg")
indexes_triangles_oldwoman, landmarks_points_face_oldwoman, face_oldwoman = swappedFace("images/OldWoman.jpg")
indexes_triangles_oldman, landmarks_points_face_oldman, face_oldman = swappedFace("images/oldman.jpg")
