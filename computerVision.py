# Die Funktionen wurden mithilfe von den Bibliotheken von Python, OpenCV geschrieben, außerdem wurde das Wissen
# aus der Vorlesung/Übung angewendet und schlussendlich für einige Hilfestellungen ChatGPT/Phind genutzt

import cv2
import numpy as np
import cv2.aruco as aruco
from scipy.spatial import distance
import sys

CAM_WIDTH, CAM_HEIGHT = 640, 480

#Sys Block
# Überprüfen, ob ein Bild oder ein Video verwendet werden soll
if len(sys.argv) > 1:
    image_filename = sys.argv[1]
    input_image = cv2.imread(image_filename)
else:
    video_capture = cv2.VideoCapture(0)
    _, input_image = video_capture.read()

# Vars
counter = 0
minDist = 100
minRadius = 50
maxRadius = 100
param1 = 100
param2 = 30
output = input_image.copy()

# Papiererkennung
if len(sys.argv) > 1:
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # Bild unscharf machen für bessere Kantenerkennung
    img_blur = cv2.GaussianBlur(img, (3,3), 0)
    # Canny-Kantenerkennung
    edges = cv2.Canny(img_blur, 75, 200)
    # Finden der Konturen
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sortieren der Konturen nach ihrer Fläche und Behalten der größten
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for contour in contours:
        # Konturumfang berechnen
        perimeter = cv2.arcLength(contour, True)
        # Kontur approximieren
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        # Wenn die approximierte Kontur vier Punkte hat, nehmen wir an, dass wir unser A4-Papier gefunden haben
        if len(approx) == 4:
            paper_contour = approx
            break
    cv2.drawContours(output, [paper_contour], -1, (0, 255, 0), 2)

    # Eckpunkte sortieren
    paper_contour = paper_contour.reshape((4, 2))
    sorted_paper_contour = paper_contour[np.argsort(paper_contour[:, 1]), :]

    # Die beiden oberen Punkte nach der x-Koordinate sortieren
    upper = sorted_paper_contour[:2]
    upper = upper[np.argsort(upper[:, 0]), :]

    # Die beiden unteren Punkte nach der x-Koordinate sortieren
    lower = sorted_paper_contour[2:]
    lower = lower[np.argsort(lower[:, 0]), :]

    # Zusammensetzen
    ordered_paper_contour = np.vstack((upper, lower[::-1]))

# Coin Detection
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 0.1, minDist=minDist,
                           param1=param1, param2=param2,
                           minRadius=minRadius, maxRadius=maxRadius)
cropped_out_coins = []
coin_diameters = []

# Circle Draw
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, radius) in circles:
        cv2.circle(output, (x, y), radius, (0, 255, 0), 3)
        cv2.circle(output, (x, y), 5, (0, 255, 0), -1)
        counter = counter + 1

        coin = output[y - radius:y + radius, x - radius:x + radius]
        cropped_out_coins.append(coin)
        coin_diameters.append(radius * 2)
# Counter

if len(sys.argv) > 1:
    # Aruco Detection
    # Erstelle ein ArUco-Wörterbuch mit einem bestimmten Code-Typ und einer Größe
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    # Definiere die Parameter für den ArUco-Detektor
    parameters = aruco.DetectorParameters()
    # Erstelle den ArUco-Detektor mit dem Wörterbuch und den Parametern
    detector = aruco.ArucoDetector(dictionary, parameters)
    # Definiere das ArUco-Wörterbuch für die spätere Verwendung
    aruco_dict = aruco.Dictionary(aruco.DICT_4X4_250, _markerSize=4)

    # Suche nach ArUco-Markern in der Ausgabebild
    corners, ids, rejectedCandidates = detector.detectMarkers(output)
    cv2.rectangle()
    # Wenn ArUco-Marker gefunden wurden, zeichne sie auf dem Ausgabebild ein
    if len(corners):
        aruco.drawDetectedMarkers(output, corners, ids)

    # Alles für die Erkennung der Münzart
    # Extrahiere die Eckpunkte des ArUco-Markers
    top_left_corner = (int(corners[0][0][0][0]), int(corners[0][0][0][1]))
    top_right_corner = (int(corners[0][0][1][0]), int(corners[0][0][1][1]))
    bottom_right_corner = (int(corners[0][0][2][0]), int(corners[0][0][2][1]))
    bottom_left_corner = (int(corners[0][0][3][0]), int(corners[0][0][3][1]))

    # Erstelle eine Vorschau des Markers auf dem Ausgabebild
    marker_preview = output.copy()
    cv2.rectangle(marker_preview, top_left_corner, bottom_right_corner, (0, 255, 0), 3)
    cv2.circle(marker_preview, top_left_corner, 20, (0, 0, 255), -1)
    cv2.circle(marker_preview, top_right_corner, 20, (255, 0, 0), -1)

    # Berechne die Breite des Markers in Pixeln
    dist = distance.euclidean(top_left_corner, top_right_corner)
    print('Width of Marker in image: {} px'.format(dist))

    # Bekannte Breite des gedruckten Markers in Millimetern
    WIDTH_OF_PRINTED_MARKER_MM = 65

    # Berechne das Verhältnis Pixel pro Millimeter
    pixel_to_mm = WIDTH_OF_PRINTED_MARKER_MM / dist
    print("=> 1 Pixel in the image equals ~{}mm".format(round(pixel_to_mm, 3)))

    # Zuordnung von bekannten Durchmessern zu Münzarten in Millimetern
    value_dict = {25.75: '2 Euro',
                  23.25: '1 Euro',
                  24.25: '50 cent',
                  22.25: '20 cent',
                  19.75: '10 cent',
                  21.25: '5 cent',
                  18.75: '2 cent',
                  16.25: '1 cent'}

    # Zuordnung von Münzarten zu Werten in Euro
    coin_dict = {'2 Euro': 2.00,
                 '1 Euro': 1.00,
                 '50 cent': 0.50,
                 '20 cent': 0.20,
                 '10 cent': 0.10,
                 '5 cent': 0.05,
                 '2 cent': 0.02,
                 '1 cent': 0.01}

    total_sum_euros = 0
    coin_types = []

    # Bestimme die Münzart für jede erkannte Münze basierend auf ihrem Durchmesser
    for i, diameter_px in enumerate(coin_diameters):
        diameter_mm = diameter_px * pixel_to_mm

        # Finde den am nächsten liegenden bekannten Durchmesser
        closest = min(value_dict.keys(), key=lambda x: abs(x - diameter_mm))
        coin_type = value_dict[closest]
        coin_types.append(coin_type)

        # Drucke den Durchmesser jeder Münze in Pixeln
        print(coin_diameters)

        # Bestimme den Wert jeder Münze in Euro und summiere sie auf
        coin_val = coin_dict[coin_type]
        total_sum_euros += coin_val

    # Zeichne die erkannten Münzen auf dem Ausgabebild ein und gib ihre Münzart aus
    for i, (x, y, radius) in enumerate(circles):
        cv2.circle(output, (x, y), radius, (0, 255, 0), 3)
        cv2.circle(output, (x, y), 5, (0, 255, 0), -1)

        coin_type = coin_types[i]
        print(coin_type)

    # Gib die Gesamtsumme der erkannten Münzen in Euro aus
    print(str(round(total_sum_euros, 2)) + "€")


    # Create a copy of the original image
    warp_image = output.copy()

    # Perspektifische Entzerrung
    # Defining the corners of the A4 paper (source points)
    pts_src = np.float32(
        [ordered_paper_contour[0], ordered_paper_contour[1], ordered_paper_contour[2], ordered_paper_contour[3]])

    # Defining the corners of the image (destination points)
    buffer = 100
    pts_dst = np.float32([[buffer, buffer],
                          [output.shape[1] - 1 - buffer, buffer],
                          [output.shape[1] - 1 - buffer, output.shape[0] - 1 - buffer],
                          [buffer, output.shape[0] - 1 - buffer]])
    # Calculating the homography matrix
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Applying the perspective transformation to the image
    warped_image = cv2.warpPerspective(output, h, (output.shape[1], output.shape[0]))
    print("Count of Coins: " + str(counter))

    blurred = cv2.GaussianBlur(warped_image, (7, 7), 0)

    # Get HSV image
    hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # For grey brick:
    lower_grey = np.array([100, 5, 120])
    upper_grey = np.array([200, 25, 180])

    # For orange brick
    lower_orange = np.array([0, 136, 140])
    upper_orange = np.array([19, 255, 255])

    # For yellow brick
    lower_yellow = np.array([22, 130, 166])
    upper_yellow = np.array([29, 255, 241])

    threshold_frame_grey = cv2.inRange(hsv_image, lower_grey, upper_grey)
    threshold_frame_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
    threshold_frame_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Count the number of contours for each color
    contours_grey, _ = cv2.findContours(threshold_frame_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_orange, _ = cv2.findContours(threshold_frame_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(threshold_frame_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 100


    def count_brick_contours(contours):
        count = 0
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                count += 1
        return count


    grey_count = count_brick_contours(contours_grey)
    orange_count = count_brick_contours(contours_orange)
    yellow_count = count_brick_contours(contours_yellow)

    # Add the first two
    threshold_frame = cv2.bitwise_or(threshold_frame_grey, threshold_frame_orange)
    # Add the third
    threshold_frame = cv2.bitwise_or(threshold_frame, threshold_frame_yellow)

    kernel = np.ones((5, 5), np.uint8)
    threshold_frame = cv2.erode(threshold_frame, kernel, iterations=3)
    threshold_frame = cv2.dilate(threshold_frame, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(warped_image, [box], 0, (255, 0, 0), 3)

    print(f"Number of grey Lego bricks: {grey_count}")
    print(f"Number of orange Lego bricks: {orange_count}")
    print(f"Number of yellow Lego bricks: {yellow_count}")

    # Display the output image with perspective correction
    cv2.imshow("Output with Perspective Correction", warped_image)
    cv2.waitKey(0)


else:
    ret, frame = video_capture.read()
    running = True
    while running:
        ret, frame = video_capture.read()
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            running = False
