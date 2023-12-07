from flask import Flask, request, make_response, render_template, send_from_directory
from werkzeug.utils import secure_filename
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv
from PIL import Image
import os
import sys
from io import BytesIO
import base64
import csv

sns.set()

# Folder to store uploads
UPLOAD_FOLDER = 'uploads'

app = Flask("yieldestimation_app",template_folder="templates")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Extensions allowed to avoid dangerous files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#################################
# Function pixelsRemoval()
# This function removes non-cotton pixels (vegetation and soil) using RGB and Lab color channels
# Inputs: RGB image
#         Lab image
# Outputs: noGreen image based on modified Excess Red and Excess Green indices
#          noSoil image based on Lab color channels
################################
def pixelsRemoval(img_RGB, img_Lab):
    # Green pixels removal
    R = img_RGB[:,:,0]
    G = img_RGB[:,:,1]
    B = img_RGB[:,:,2]
    #calculate normalized color channels
    Rnorm = cv.normalize(R, None, 0, 1, cv.NORM_MINMAX, cv.CV_32FC1)
    Gnorm = cv.normalize(G, None, 0, 1, cv.NORM_MINMAX, cv.CV_32FC1)
    Bnorm = cv.normalize(B, None, 0, 1, cv.NORM_MINMAX, cv.CV_32FC1)
    
    #original ExR and ExG indices
    #ExR = 1.4*Rnorm - Gnorm;
    #ExG = 2*Gnorm - Rnorm - Bnorm;
    #optimum cotton pixels preprocessing (modified Excess Red and Excess Green indices)
    ExR = 2.0*Rnorm - Gnorm;
    ExG = 1.5*Gnorm - Rnorm - Bnorm;
    
    ExR = cv.normalize(ExR, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    #thR, ExR = cv.threshold(ExR, 0, 255, cv.THRESH_OTSU);
    ExR = ExR.astype(np.int16)

    ExG = cv.normalize(ExG, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    #thG, ExG = cv.threshold(ExG, 0, 255, cv.THRESH_OTSU);
    ExG = ExG.astype(np.int16)

    mask_noVegetation = cv.normalize(cv.normalize(np.subtract(ExG,ExR) * -1, 
                            None, 0, 1, cv.NORM_MINMAX) * 255, None, 0,
                            255, cv.NORM_MINMAX,cv.CV_8U)
    #img_Green = cv.bitwise_not(img_noGreen)

    rgb_noVegetation = cv.bitwise_and(img_RGB,img_RGB,mask = mask_noVegetation)
    #rgb_green = cv.bitwise_and(img,img,mask = mask_Green)

    #bgr_noGreen = cv.cvtColor(rgb_noGreen, cv.COLOR_RGB2BGR)
    #display(Image.fromarray(rgb_ground))
    
    #Soil pixels removal
    Lab_noVegetation = cv.bitwise_and(img_Lab,img_Lab,mask = mask_noVegetation)
    #color space splitting for soild extraction
    L = Lab_noVegetation[:,:,0]
    a = Lab_noVegetation[:,:,1]
    b = Lab_noVegetation[:,:,2]
    #calculate normalized color channels
    Lnorm = cv.normalize(L, None, 0, 1, cv.NORM_MINMAX, cv.CV_32FC1)
    anorm = cv.normalize(a, None, 0, 1, cv.NORM_MINMAX, cv.CV_32FC1)
    bnorm = cv.normalize(b, None, 0, 1, cv.NORM_MINMAX, cv.CV_32FC1)
    
    ExL = 1.5*Lnorm - anorm;
    Exa = 1*anorm - Lnorm - bnorm;
    
    ExL = cv.normalize(ExL, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    #thL, ExL = cv.threshold(ExL, 0, 255, cv.THRESH_OTSU);
    ExL = ExL.astype(np.int16)

    Exa = cv.normalize(Exa, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    #tha, Exa = cv.threshold(Exa, 0, 255, cv.THRESH_OTSU);
    Exa = Exa.astype(np.int16)

    mask_noSoil = cv.normalize(cv.normalize(np.subtract(Exa,ExL) * -1, 
                            None, 0, 1, cv.NORM_MINMAX) * 255, None, 0,
                            255, cv.NORM_MINMAX,cv.CV_8U)
    #mask_Soil = cv.bitwise_not(mask_noSoil)

    rgb_noSoil = cv.bitwise_and(img_RGB,img_RGB,mask = mask_noSoil)
    #rgb_Soil = cv.bitwise_and(img_Soil,RGB,mask = mask_Soil)

    #bgr_noSoil = cv.cvtColor(rgb_noSoil, cv.COLOR_RGB2BGR)
    #display(Image.fromarray(rgb_noSoil))
    
    return(rgb_noVegetation, rgb_noSoil, mask_noVegetation, mask_noSoil)

#################################
# Function colorSpaces()
# This function transform an input image into 3 different color spaces (RGB, HSV, CIELa*b*)
# Inputs: BGR image
# Outputs: RGB image
#          HSV image
#          CIE Lab image
################################
#This generic function takes a BGR image as input and returns 3 different images:
#RGB, HSV, CIE Lab color spaces.
def colorSpaces(img):
    #print("you are here! coloSpaces")
    # Convert BGR to RGB
    RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #display(Image.fromarray(RGB))
    #R,G,B = cv.split(RGB)
    # Convert BGR to HSV
    HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #display(Image.fromarray(HSV))
    #H,S,V = cv.split(HSV)
    # Convert BGR to CIELab
    Lab=cv.cvtColor(img,cv.COLOR_BGR2LAB)
    #display(Image.fromarray(Lab))
    #L,a,b = cv.split(Lab)
    
    return (RGB,HSV,Lab)

#################################
# Function createValidationDF()
# This function creates the dataset for SVM model clasification
# Inputs: RGB image
#         HSV image
#         Lab image
# Outputs: Full datafrrame containing all pixels in the final image
#          Dataframe containing only cotton related pixels
################################
# It requires the HSV and Lab color space images as inputs, and it returns the daframe (df) for validation.
# This dataset includes just H, S, V, a, b color channel components.
# Annotated masks are not needed.

def createValidationDF(RGB,HSV,Lab):
    # Create RGB subset
    index = pd.MultiIndex.from_product(
        (*map(range, RGB.shape[:2]), ('R', 'G', 'B')),
        names=('row', 'col', None))
    # Can be chained but separated for use in explanation
    df1 = pd.Series(RGB.flatten(), index=index)
    df1 = df1.unstack()
    df1 = df1.reset_index().reindex(columns=['row', 'col', 'R', 'G', 'B'])
    
    #Create HSV subset
    index = pd.MultiIndex.from_product(
        (*map(range, HSV.shape[:2]), ('H', 'S', 'V')),
        names=('row', 'col', None))
    # HSV color channels extraction
    df2 = pd.Series(HSV.flatten(), index=index)
    df2 = df2.unstack()
    df2 = df2.reset_index().reindex(columns=['row', 'col', 'H', 'S', 'V'])
    
    #Create CIE Lab subset
    index = pd.MultiIndex.from_product(
        (*map(range, Lab.shape[:2]), ('L', 'a', 'b')),
        names=('row', 'col', None))
    # Lab color channels extraction
    df3 = pd.Series(Lab.flatten(), index=index)
    df3 = df3.unstack()
    df3 = df3.reset_index().reindex(columns=['row', 'col', 'L', 'a', 'b'])

    # Merge RGB, HSV and Lab subsets using pixel location (row, col)
    df = pd.merge(df1, df2, on=['row', 'col'])
    df = pd.merge(df, df3, on=['row', 'col'])

    #reduce number of points by deleting non-cotton pixels from dataset (R=G=B=H=S=V=L=a=b=0)
    df_temp = df.copy()
    df_cotton = df_temp[(df_temp['R'] > 0) & (df_temp['G'] > 0) & (df_temp['B'] > 0) & (df_temp['H'] > 0) & (df_temp['S'] > 0) & (df_temp['V'] > 0) & (df_temp['L'] > 0) & (df_temp['a'] > 0) & (df_temp['b'] > 0)] 
    
    return df, df_cotton

# Function that gets the current figure as a base 64 image for embedding into websites
def getCurrFigAsBase64HTML(fig):
    im_buf_arr = BytesIO()
    fig.savefig(im_buf_arr,format='png',bbox_inches='tight')
    im_buf_arr.seek(0)
    b64data = base64.b64encode(im_buf_arr.read()).decode('utf8');
    return b64data
    #return render_template('img.html',img_data=b64data)

# Function to store data into the dataset
def add_data(data_in):
    global data_df
    plt.clf()
    # Store predicted pixel number into the csv file
    with open('data.csv', 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj,delimiter=',',lineterminator='\n')
        csv_writer.writerow([data_in.loc[0]["plot_id"], data_in.loc[0]["boll number"]])
    data_df = pd.concat([data_df,data_in],ignore_index=True)
    fig2 = sns.barplot(x = "plot_id", y = "boll number", data = data_df, color='green')
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    fig2.set(xlabel="Plot ID", ylabel = "Cotton bolls")
    fig=plt.gcf()
    img_data = getCurrFigAsBase64HTML(fig);
    # render dataframe as html
    data_html = data_df.to_html()
    return render_template('done.html', table=data_html, graph=img_data)

def init():
    global data_df
    data_df = pd.read_csv('data.csv')

try:
    # Read data from csv 
    data_df = pd.read_csv('data/data.csv')
    # Load SVM classification model
    clf = joblib.load('model/SVM_model_BHSb_C_1_gamma_Scale_noSoil.pkl')
    #print("model loaded successfully!")

except:
    init()

# This method resets/initializes the database when called. It is password protected
@app.route("/reset")
def reset():
    return render_template("reset.html")
# Method asking for reset confirmation (it requires user and password)
@app.route("/reset_confirmation", methods=["GET"])
def reset_confirmation():
    if request.authorization and request.authorization.username == "user" and request.authorization.password == "pass":
        # Delete database content using w+ (truncate file to 0 size)
        f = open('data.csv', "w+")
        f.close()
        # Initialize database headers
        with open('data.csv', 'w+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = csv.writer(write_obj,delimiter=',',lineterminator='\n')
            # Add contents of list as last row in the csv file
            csv_writer.writerow(['plot_id','boll number'])
        init()
        return "Database reset"
    return make_response("Unauthorized Access", 401, {'WWW-Authenticate' : 'Basic realm="Login Required"'})

# Show main screen
@app.route("/")
def main():
    return render_template("main.html")

# Redirect to the interface associated to the selected method (manual or image analysis)
@app.route("/data_input_method", methods=["POST"])
def data_input_method():
    global method
    method1 = request.form["method"]
    if method1 == "Analysis":
        return render_template("imgAnalysis.html")
    else:
        return render_template("manualInput.html",data1="0",data2="0") 

# Show interface to enter data manually
@app.route("/data_input", methods=["POST"])
def data_input():
    global newdata_df
    plot_name = request.form["plot_name"]
    new_data = request.form["new_data"]
    new_data_msg =plot_name+', '+new_data
    newdata_df = pd.DataFrame([[plot_name,int(new_data)]],columns=['plot_id','boll number'])
    msg = add_data(newdata_df)
    return msg

# Store uploaded image
#@app.route("/uploads/<filename>")
#def uploaded_file(filename):
#    return send_from_directory(app.config["UPLOAD_FOLDER"],
#                               filename)

# Previsualize image for confirmation
@app.route("/img_upload", methods=["POST"])
def upload_file():
    global newdata_df
    global file
    global image
    global image_web
    # check if the image has been included in the request
    if "file" not in request.files:
        return error
    file = request.files['file']
    # check if the image exists
    if file.filename == '':
        return error

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Extract the name of the file including its extension
        fileName = os.path.basename(file.filename)
        # Use splitext() to get filename and extension separately.
        (name, ext) = os.path.splitext(fileName)
        # Save uploaded image
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # and read it
        image = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img_ = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        fig0 = plt.imshow(img_)
        plt.axis('off')
        img_ = plt.gcf()
        image_web = getCurrFigAsBase64HTML(img_)

        return render_template("imgConfirmation.html", data1=name ,img=image_web)
    return error

# Process image
@app.route("/img_confirmation", methods=['POST'])
def img_confirmation():
    global file
    global image
    global image_web

    # Read name from form
    plot_name = request.form["plot_name_conf"]

    # Extract the name of the file including its extension
    fileName = os.path.basename(file.filename)
    # Use splitext() to get filename and extension separately.
    (name, ext) = os.path.splitext(fileName)

    # Store original image
    #cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], (plot_name + ext)), image);

    # Read image
    #img = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], plot_name + ext))
    #if img == None:
    #print('no image to read! Check path: ' + os.path.join(app.config['UPLOAD_FOLDER'], (plot_name + ext)), file=sys.stderr)
    #else:
    #    print('image read successfully!', file=sys.stderr)

    # Transform color spaces (it expects a BGR image)
    RGB,HSV,Lab = colorSpaces(image)
    
    # Extract plant and soil pixels
    RGB_noVegetation,RGB_noSoil,mask_noVegetation,mask_noSoil = pixelsRemoval(RGB,Lab)

    # Create dataframe
    RGB_noVegetation = cv.bitwise_and(RGB,RGB,mask = mask_noVegetation)
    RGB_noSoil = cv.bitwise_and(RGB_noVegetation,RGB_noVegetation,mask = mask_noSoil)
    #display(Image.fromarray(RGB_noSoil))
    HSV_noVegetation = cv.bitwise_and(HSV,HSV,mask = mask_noVegetation)
    HSV_noSoil = cv.bitwise_and(HSV_noVegetation,HSV_noVegetation,mask = mask_noSoil)
    #display(Image.fromarray(HSV_noSoil))
    Lab_noVegetation = cv.bitwise_and(Lab,Lab,mask = mask_noVegetation)
    Lab_noSoil = cv.bitwise_and(Lab_noVegetation,Lab_noVegetation,mask = mask_noSoil)

    # Create dataset for validation
    full_data, cotton_data = createValidationDF(RGB_noSoil,HSV_noSoil,Lab_noSoil)

    # SVM pixels classification
    # Drop pixel position columns (row, and col)
    X = cotton_data.drop(['row','col','R','G','V','L','a'], axis = 1)
    
    # Apply SVM model for pixels classification    
    y_pred = clf.predict(X)

    # Create prediction mask
    cotton_df = cotton_data.copy()

    # Recover pixel indices
    cotton_df.insert(0,'cotton', y_pred, True)
    cotton_df = pd.DataFrame(cotton_df.iloc[:,0])
    cotton_df[cotton_df['cotton']=='no'] = 0
    cotton_df[cotton_df['cotton']=='yes'] = 255

    # Reconstruct image dataframe
    full_mask = full_data.copy()
    full_mask['cotton']=0
    full_mask = pd.DataFrame(full_mask.iloc[:,-1])
    #substitute values with cotton prediction values
    full_mask.update(cotton_df)
    # Change dtype to 'uint8'
    full_mask = (full_mask.astype('uint8')).to_numpy()
    # Convert from array to image pixels
    h,w,c = RGB_noSoil.shape
    mask_result = np.reshape(full_mask,(h,w))

    # Count number of cotton pixels found in the image
    cottonPx_number = np.count_nonzero(mask_result)
    print(cottonPx_number)

    # Create cotton mask image
    cotton_mask = cv.cvtColor(mask_result,cv.COLOR_GRAY2RGB)
    print(np.count_nonzero(cotton_mask))

    # Save mask
    cotton_filename = os.path.splitext(file.filename)[0] + '_cotton' + os.path.splitext(file.filename)[1]
    cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], cotton_filename),cotton_mask)
    #cotton_img = Image.fromarray(cotton_mask)
    #cotton_img.save(os.path.join(app.config['UPLOAD_FOLDER'], cotton_filename))
    
    # Image masks post-processing
    # Read cotton mask image
    mask = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], cotton_filename))
    #print(np.count_nonzero(mask))

    # Define morphological elements
    se_erode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    se_dilate = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    
    # Apply erosion
    erodedBW= cv.erode(mask, se_erode, iterations=1)
    #print(np.count_nonzero(erodedBW))
    # Apply dilation
    dilatedBW = cv.dilate(erodedBW, se_dilate, iterations=1)
    #print(np.count_nonzero(dilatedBW))
#    cotton_img = Image.fromarray(dilatedBW)
#    cotton_img.save(os.path.join(app.config['UPLOAD_FOLDER'], cotton_filename))

    # Compute pixel clusters
    CC = cv.connectedComponentsWithStats(dilatedBW[:,:,0], 8)
    num_labels = CC[0]
    labels = CC[1]
    centroids = CC[3]
    height =  cotton_mask.shape[0]
    width =  cotton_mask.shape[1]
    
    componentMask = np.zeros((height,width,3), np.uint8)
       
    # Store number of clusters (cotton bolls)
    boll_number = num_labels
    print(num_labels)
    
    # Overlay detected cotton clusters on original image

    # Store resulting cotton clusters mask
#    cotton_filename = os.path.splitext(file.filename)[0] + '_cotton.png'
#    cotton_img = Image.fromarray(componentMask)
    #cotton_img.save(os.path.join(app.config['UPLOAD_FOLDER'], cotton_filename))

    # Read original RGB image again
#    img_ = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], cotton_filename))
    # Convert to gray scale    
#    img_gray = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
    
    image_ = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    imageCenters = image_.copy()

    # Draw clusters
    cmap = plt.get_cmap('jet',num_labels)
    colors = cmap(range(num_labels))
    np.random.seed(42)    
    for i in range(num_labels):
        if i == 0:
            continue
        # Draw cluster centers
        centers = cv.circle(imageCenters, (int(centroids[i,0]),int(centroids[i,1])), 2, (255,0,0), -1)
        color = np.uint8(colors[np.random.random_integers(0, num_labels-1,1)]*255)
        # Create color masks
        componentMask[labels == i] = color[0,:3]
    # Overlay masks
    overlay = cv.addWeighted(image_, 1, componentMask, 0.9, 0)
 
    # Store resulting image masks
    centers_filename = os.path.splitext(file.filename)[0] + '_centers' + os.path.splitext(file.filename)[1]
    masks_filename = os.path.splitext(file.filename)[0] + '_masks' + os.path.splitext(file.filename)[1]

    centers_img = Image.fromarray(centers)
    overlay_img = Image.fromarray(overlay)

    centers_img.save(os.path.join(app.config['UPLOAD_FOLDER'], centers_filename))
    overlay_img.save(os.path.join(app.config['UPLOAD_FOLDER'], masks_filename))

    fig2 = plt.imshow(overlay)
    plt.axis('off')
    img_ = plt.gcf()
    image_web2 = getCurrFigAsBase64HTML(img_)

    plot_data = os.path.splitext(file.filename)[0]+ "," + str(boll_number)
    newdata_df = pd.DataFrame([[os.path.splitext(file.filename)[0],boll_number]],columns=['plot_id','boll number'])

    return render_template("dataConfirmation.html", data1=os.path.splitext(file.filename)[0], data2=str(boll_number), img_orig=image_web, img=image_web2)

# Show interface to confirm data automatically extrated from the uploaded image
@app.route("/data_confirmation", methods=["POST"])
def data_confirmation():
    global newdata_df
    plot_name = request.form["plot_name_conf"]
    new_data = request.form["new_data_conf"]
    new_data_msg =plot_name+', '+new_data
    newdata_df = pd.DataFrame([[plot_name,int(new_data)]],columns=['plot_id','boll number'])
    msg = add_data(newdata_df)
    return msg

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)