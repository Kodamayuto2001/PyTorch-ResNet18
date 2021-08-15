import cv2

img = cv2.imread("dataset/renamed/721.jpg")

img = cv2.resize(img,(160,160))

cv2.imshow("0",img)

if cv2.waitKey(10000) & 0xFF == ord("q"):
    pass


