import cv2

video = cv2.VideoCapture("data/mercadinho.mp4")

while True:
    ret, frame = video.read()
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord("s"):  # press s for save frame
        cv2.imwrite(f"data/imgs_test/frame_mercadinho.jpg", frame)
        print("Captura salva com sucesso!")
        
        
    if key == ord("q"):
        break
    
video.release()
cv2.destroyAllWindows()