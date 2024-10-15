import cv2
import streamlit as st
import numpy as np
from PIL import Image
import torch
from typing import List
from torchvision import transforms
from PIL import Image
import imghdr
import imutils
import math
import pandas as pd
from streamlit_js_eval import streamlit_js_eval

def convert2png(image):
    image_type = imghdr.what(image)
    if image_type == 'png':
        converted_image = Image.open(image)
        converted_image = np.array(converted_image)
    elif image_type == 'jpeg':
        converted_image = Image.open(image)
        converted_image = np.array(converted_image)
    else:
        return image
    return converted_image
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

def sheet2bottom(sheet):
    gamelog_bounding_points = []
    for (image_path,min_factor,max_factor) in zip(
        ['results-6.35-6.45.png', 'remarks-1-1.1.png'],
        [6.35,1],[6.45,1.1]):
        template = cv2.imread(image_path)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        (tH, tW) = template.shape[:2]

        gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
        found = None
        linspace_var = max(template.shape[0]/sheet.shape[0],template.shape[1]/sheet.shape[1])
        for scale in np.linspace(min_factor*linspace_var, max_factor*linspace_var, 20)[::-1]:
            resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            if resized.shape[0] < tH or resized.shape[1] < tW:
                continue
            edged = cv2.Canny(resized, 50, 200)
            result   = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            if found is None or maxVal > found[0]: found = (maxVal, maxLoc, r)

        (maxVal, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        gamelog_bounding_points.append((startX, startY, endX, endY))

    return sheet[gamelog_bounding_points[1][3]:, :gamelog_bounding_points[0][0]]

def get_gamelog_corner_pts(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(np.array(binary).shape[0]/2)))
    temp = cv2.dilate(binary, np.ones((1,3)), iterations=1)
    vertical_lines_img = cv2.morphologyEx(temp, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(np.array(binary).shape[1]/20), 1))
    temp = cv2.dilate(binary, np.ones((3,1)), iterations=1)
    horizontal_lines_img = cv2.morphologyEx(temp, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    table = cv2.bitwise_or(vertical_lines_img, horizontal_lines_img)
    canny = cv2.Canny(table, 0.5, 1, None, 3)
    contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    corner_pts = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        corner_pts.append([(x,y),(x+w,y),(x,y+h),(x+w,y+h)])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    corner_pts = sorted(corner_pts,key=lambda data:data[0][0]*data[0][0]+data[0][1]*data[0][1])
    return corner_pts

def get_gamelogs(img,corner_pts):
    gamelog_height = 500
    gamelog_width = 369
    gamelog_template_corners = np.array([[0,0], [gamelog_width,0], [0,gamelog_height], [gamelog_width,gamelog_height]])

    gamelogs = []
    for i in range(5):
        gamelog_corners = np.array(corner_pts[i])
        (H, mask) = cv2.findHomography(gamelog_corners, gamelog_template_corners, method=cv2.RANSAC)
        gamelog = cv2.warpPerspective(img, H, (gamelog_width, gamelog_height))
        gamelogs.append(gamelog)
    return gamelogs

def draw_lines_in_gamelog(img,number_of_empty_rows,column_x_percent):
    gamelog = np.copy(img)
    gamelog_height = gamelog.shape[0]
    gamelog_width = gamelog.shape[1]
    row_y_values = [int(gamelog_height/(number_of_empty_rows+1)*(i+1)) for i in range(number_of_empty_rows+1)]
    column_x_values = [int(i/100*gamelog_width) for i in column_x_percent]
    for y1 in row_y_values[:-1]:
        y2 = row_y_values[row_y_values.index(y1)+1]
        for x1 in column_x_values[:-1]:
            x2 = column_x_values[column_x_values.index(x1)+1]
            cv2.rectangle(gamelog, (x1, y1), (x2, y2), (255,0,0), 2)
    return gamelog

def get_cell_imgs(gamelog,number_of_empty_rows,column_x_percent):
    gamelog_height = gamelog.shape[0]
    gamelog_width = gamelog.shape[1]
    row_y_values = [int(gamelog_height/(number_of_empty_rows+1)*(i+1)) for i in range(number_of_empty_rows+1)]
    column_x_values = [int(i/100*gamelog_width) for i in column_x_percent]

    row_img=[]; cap_img=[]; team_img=[]; remarks_img=[]
    gamelog = image2table(gamelog,kind='extract_data')
    for y1 in row_y_values[:-1]:
        y2 = row_y_values[row_y_values.index(y1)+1]
        row_img.append( gamelog[y1:y2,0:gamelog_width] )
        for (x1,col_imgs) in zip(column_x_values[1:4],[cap_img,team_img,remarks_img]):
            x2 = column_x_values[column_x_values.index(x1)+1]
            col_imgs.append( gamelog[y1:y2, x1:x2] )
    return row_img,cap_img,team_img,remarks_img

def preprocess_row(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (3,3), 0)
    binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    dilated = cv2.dilate(binary, np.ones((1,1),np.uint8), iterations=1)
    output = cv2.resize(dilated, (28*11,28))
    return output

def intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(x0.item()), int(y0.item())
    return [x0, y0]

def preprocess(image, pct_offset=0.15, cap_column=False, offset=0):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (3,3), 0)
    binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    binary_simplified = np.copy(binary)
    if cap_column:
        output = preprocess_cap(binary_simplified)
    else:
        cols_to_delete = []
        for col in range(binary_simplified.shape[1]):
            num_white_pixels = 0
            for pixel in binary_simplified[int(binary_simplified.shape[1]*pct_offset):int(binary_simplified.shape[1]*(1-pct_offset)),col]:
                if pixel > 0: num_white_pixels += 1
            if num_white_pixels < 3: cols_to_delete.append(col)
        for col in reversed(cols_to_delete):
            binary_simplified = np.delete(binary_simplified, col, 1)
        if binary_simplified.shape[1] < 10:
            binary_simplified = np.zeros((binary_simplified.shape[0],binary_simplified.shape[0]))
        
        output = np.copy(binary_simplified)
        border = abs(output.shape[1] - output.shape[0]) / 2
        if output.shape[1] > output.shape[0]:
            output = cv2.copyMakeBorder(binary_simplified, math.floor(border), math.ceil(border), 0, 0, cv2.BORDER_CONSTANT)
        else:
            output = cv2.copyMakeBorder(binary_simplified, 0, 0, math.floor(border), math.ceil(border), cv2.BORDER_CONSTANT)
        output = cv2.resize(output, (28,28))

    return output

def preprocess_cap(binary_simplified):
    cols_to_delete = []
    edge_columns = list(range(int(binary_simplified.shape[1]*0.2))) + list(range(int(binary_simplified.shape[1]*0.8),binary_simplified.shape[1]))
    for col in edge_columns:
        num_white_pixels = 0
        for pixel in binary_simplified[int(binary_simplified.shape[1]*0.10):int(binary_simplified.shape[1]*0.90),col]:
            if pixel > 0: num_white_pixels += 1
        if num_white_pixels < 3: cols_to_delete.append(col)
    for col in reversed(cols_to_delete):
        binary_simplified = np.delete(binary_simplified, col, 1)
    if binary_simplified.shape[1] < 10:
        binary_simplified = np.zeros((binary_simplified.shape[0],binary_simplified.shape[0]))

    output = np.copy(binary_simplified)
    border = abs(output.shape[1] - output.shape[0]*40/28) / 2
    if output.shape[1] > output.shape[0]*40/28:
        output = cv2.copyMakeBorder(binary_simplified, math.floor(border), math.ceil(border), 0, 0, cv2.BORDER_CONSTANT)
    else:
        output = cv2.copyMakeBorder(binary_simplified, 0, 0, math.floor(border), math.ceil(border), cv2.BORDER_CONSTANT)
    output = cv2.resize(output, (40,28))

    return output

def predict_cell(   model: torch.nn.Module,
                    image: np.array,
                    class_names: List[str] = None):

    transform = transforms.ToTensor()
    img = Image.fromarray(image)
    input = transform(img)
    input = input.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    image_pred = model(input)

    image_pred_probs = torch.softmax(image_pred, dim=1) # convert logits --> prediction probabilities
    image_pred_label = torch.argmax(image_pred_probs, dim=1) # convert prediction probabilities --> prediction labels

    if class_names:
        prediction = class_names[image_pred_label.cpu()]
        probability = image_pred_probs.max().cpu()
        return prediction
    else:
        prediction = image_pred_label.cpu()
        probability = image_pred_probs.max().cpu()
        return prediction

def image2table(image,kind='normal'):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(np.array(binary).shape[0]/20)))
    tempv = cv2.dilate(binary, np.ones((1,3)), iterations=1)
    vertical_lines_img = cv2.morphologyEx(tempv, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    tempv1 = cv2.erode(vertical_lines_img, vertical_kernel, iterations = 1)
    extended_cols = cv2.morphologyEx(vertical_lines_img, cv2.MORPH_CLOSE, vertical_kernel, iterations=3)
    extended_cols = cv2.dilate(extended_cols, vertical_kernel, iterations=5)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(np.array(binary).shape[1]/20), 1))
    temph = cv2.dilate(binary, np.ones((3,1)), iterations=1)
    horizontal_lines_img = cv2.morphologyEx(temph, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    temph1 = cv2.erode(horizontal_lines_img, horizontal_kernel, iterations=1)
    extended_rows = cv2.morphologyEx(horizontal_lines_img, cv2.MORPH_CLOSE, horizontal_kernel, iterations=3)

    bin_wo_table = binary - cv2.medianBlur(tempv1+temph1,ksize=5)
    table = cv2.bitwise_or(vertical_lines_img, horizontal_lines_img)

    if kind == 'normal': return table
    if kind == 'extended_rows': return extended_rows
    if kind == 'extended_cols': return extended_cols
    if kind == 'extract_data': return 255 - cv2.cvtColor(bin_wo_table, cv2.COLOR_GRAY2BGR)

def analyze_rows(raw_bottom):
    table = image2table(raw_bottom,kind="extended_rows")
    edged = cv2.Canny(table, 50, 200, None, 3)
    for threshold in np.linspace(100,2000,10):
        lines = cv2.HoughLines(table,1,np.pi/180,int(threshold))
        if len(lines) < 17: break
    lines = [line for line in lines if line[0][0] > 0]
    rhos = [line[0][0] for line in lines]
    return min(rhos), max(rhos), len(lines)-2

def analyze_cols(raw_bottom):
    table = image2table(raw_bottom,kind="extended_cols")
    edged = cv2.Canny(table, 50, 200, None, 3)
    for threshold in np.linspace(1000,2000,50):
        lines = cv2.HoughLines(table,1,np.pi/180,int(threshold))
        lines = sorted(lines, key=lambda x: x[0][0])
        i = 0
        while i < len(lines)-1:
            if abs( lines[i][0][0] - lines[i+1][0][0] ) < 60: lines.pop(i+1)
            else: i += 1
        if lines is not None and len(lines) == 30: break
    rhos = [line[0][0] for line in lines]
    return np.array(rhos).reshape(5,6)

def main_loop():
    DEBUG = False


    st.set_page_config(page_title="Water Polo Stats", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 400px;
            margin-left: -400px;
        }
        
        """,
        unsafe_allow_html=True,
    )

    st.title("CWPA Scoresheet Statistics Extractor")

    screen_width = streamlit_js_eval(js_expressions='screen.width', key = 'SCR')

    image_file = st.file_uploader("Upload Your Image", type=['jpg','png','jpeg'])
    if not image_file: return None

    left_col, right_col = st.columns([3,1])

    with left_col:
        sheet = convert2png(image_file)
        sheet = cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)
        st.image(sheet)

        sheet_bottom = sheet2bottom(sheet)
        (Ytop,Ybottom,number_of_empty_rows) = analyze_rows(sheet_bottom)
        number_of_empty_rows = 13
        Xcolumns = analyze_cols(sheet_bottom)
        corner_pts = []
        for i in range(5):
            corner_pts += [[
                (int(Xcolumns[i][0]),int(Ytop)),
                (int(Xcolumns[i][-1]),int(Ytop)),
                (int(Xcolumns[i][0]),int(Ybottom)),
                (int(Xcolumns[i][-1]),int(Ybottom))]]
        
        gamelogs = get_gamelogs(sheet_bottom,corner_pts)

        if DEBUG:
            for (pt1,pt2,pt3,pt4) in corner_pts:
                cv2.circle(sheet_bottom,pt1,20,(255,0,0),-1)
                cv2.circle(sheet_bottom,pt2,20,(255,255,0),-1)
                cv2.circle(sheet_bottom,pt3,20,(0,255,255),-1)
                cv2.circle(sheet_bottom,pt4,20,(255,0,255),-1)
            st.image(sheet_bottom)

            gamelogs_w_lines = []
            for (i,gamelog) in enumerate(gamelogs):
                column_x_pcts = 100*(Xcolumns[i][:]-Xcolumns[i][0])/(Xcolumns[i][-1]-Xcolumns[i][0])
                st.text(column_x_pcts)
                gamelog_w_lines = draw_lines_in_gamelog(gamelog,number_of_empty_rows,column_x_pcts)
                gamelogs_w_lines.append(gamelog_w_lines)
            st.image(gamelogs_w_lines,width=int(screen_width/9))

    row_imgs=[]; cap_imgs=[]; team_imgs=[]; remarks_imgs=[]
    for i in range(5):
        column_x_pcts = 100*(Xcolumns[i][:]-Xcolumns[i][0])/(Xcolumns[i][-1]-Xcolumns[i][0])
        row_img,cap_img,team_img,remarks_img = get_cell_imgs(gamelogs[i],number_of_empty_rows,column_x_pcts)
        row_imgs+=row_img; cap_imgs+=cap_img; team_imgs+=team_img; remarks_imgs+=remarks_img

    models_dict = {}
    for model in ['row','cap','team','remarks']:
       models_dict[model] = torch.jit.load(f'{model}_model.pt',map_location=torch.device('cpu'))

    quarter = 1
    table = [['ROW','CAP','TEAM','REMARKS']]
    for i in range(len(row_imgs)):
        row_processed = preprocess_row(row_imgs[i])
        empty_or_filled = predict_cell(model=models_dict['row'],
                                      image=row_processed,
                                      class_names=['empty','filled'])
        if empty_or_filled == 'empty':
            table.append([i+1,' ----- ',' ----- ',' ----- '])
            quarter = min(quarter+1,5)
            continue
        
        cap_processed = preprocess(cap_imgs[i], cap_column=True)
        cap = predict_cell(model=models_dict['cap'],
                           image=cap_processed,
                           class_names=[str(i) for i in range(30)])
        
        team_processed = preprocess(team_imgs[i], pct_offset=0.2)
        team = predict_cell(model=models_dict['team'],
                            image=team_processed,
                            class_names=['D','W'])

        remarks_processed = preprocess(remarks_imgs[i])
        remarks = predict_cell(model=models_dict['remarks'],
                               image=remarks_processed,
                               class_names=['C-card','E-ejection','G-goal','P-penalty','TO-timeout'])
        if remarks == 'TO-timeout': cap = ''

        table.append([i+1, cap, team, remarks])
    i += 1
    while(table[i][2] == ' ----- '):
        table.pop(i)
        i -= 1

    right_col.data_editor(
        pd.DataFrame(table[1:], columns=table[0]),
        height=int(screen_width/2),
        use_container_width=True,
        num_rows="dynamic"
    )

if __name__ == '__main__':
    main_loop()