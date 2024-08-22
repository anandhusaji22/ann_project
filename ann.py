#%%
import cv2
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0) 


def showimg(img, cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()


def resize(img):
    target_size = (28, 28)

    # Get the original dimensions
    original_height, original_width = img.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    # Determine the new size while maintaining the aspect ratio
    if aspect_ratio > 1:  # Wider than tall
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:  # Taller than wide or square
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)

    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Calculate padding to be added
    pad_top = (target_size[1] - new_height) // 2
    pad_bottom = target_size[1] - new_height - pad_top
    pad_left = (target_size[0] - new_width) // 2
    pad_right = target_size[0] - new_width - pad_left

    # Add padding to the resized image to make it 28x28
    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def symbol_position(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(image, 153, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('blacky', binary)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum area threshold (adjust this value based on your needs)
    min_area = 5

    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    symbols_loc = [cv2.boundingRect(contour) for contour in filtered_contours]
    symbols_loc.sort(key= lambda x : x[0] )
    return symbols_loc,binary



#%%
def vedioTracking():
    while True:
        _, frame = cap.read() 

        # Flip image 
        # frame = cv2.flip(frame, 1) 
        # Draw a rectangle on the frame
        x, y, w, h = 200, 200, 200, 100  # Example values (x, y) is top-left and (w, h) is width and height
        cropped_frame = frame[y:y+h, x:x+w].copy()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)
        # Process the image to get the position and binary mask
        pos, binary = symbol_position(cropped_frame)

        # Display text at the specified position
        text = "=3"
        if pos:
            x_pos, y_pos,w_pos,h_pos = pos[-1]  # Assuming `pos` contains the (x, y) position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 0, 0)  # White color
            thickness = 2
            # Draw the text on the image
            cv2.putText(frame, text, (x_pos+x+w_pos+20, y_pos+y+h_pos), font, font_scale, color, thickness)
            for p in pos:
                conx,cony,conW,conH = p
                cv2.rectangle(frame,(conx+x,cony+y),(conx+x+conW,cony+y+conH),(255, 0, 0),1)



        
        # Optionally, display the frame with the text (e.g., in a window)
        cv2.imshow('Frame with Text', frame)
        if cv2.waitKey(1) & 0xff == ord('q'): 
            break
    cap.release()
    cv2.destroyAllWindows()

#%%
vedioTracking()
# image = cv2.imread(r'C:\Mathlab\my code\ANN\annpro\ann_project\img.jpg')
# pos,binary=symbol_position(image)
# symbols = [resize(binary[y:y+h, x:x+w]) for x, y, w, h in pos]

# count=1
# f = "C:\\Mathlab\\my code\\ANN\\annpro\\ann_project\\images\\"
# for sym in symbols:
#     showimg(sym)
#     cv2.imwrite(f'{f}\symbol{count}.png', sym)
#     count+=1
