from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)

# Lấy đường dẫn đến thư mục hiện tại (nơi chứa app.py)
current_directory = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn đến thư mục 'static/output' bên trong 'Tranfrom_Image'
OUTPUT_FOLDER = os.path.join(current_directory, 'static', 'output')

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    
# Hàm xóa tất cả các tệp trong thư mục output
def clear_output_folder():
    for filename in os.listdir(OUTPUT_FOLDER):
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Không thể xóa tệp {file_path}: {e}")

# Hàm áp dụng các bộ lọc
def apply_filters(image):
    # Bộ lọc phác thảo
    def sketch_filter(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        invert_image = cv2.bitwise_not(gray_image)
        blur = cv2.GaussianBlur(invert_image, (21, 21), 0)
        inverted_blur = cv2.bitwise_not(blur)
        return cv2.divide(gray_image, inverted_blur, scale=256.0)

    # Bộ lọc sơn dầu
    def oil_paint_filter(image):
        return cv2.xphoto.oilPainting(image, 7, 1)

    # Bộ lọc làm mờ
    def blur_filter(image):
        return cv2.GaussianBlur(image, (35, 35), 0)

    # Bộ lọc tranh ghép (Mosaic)
    def mosaic_filter(image, scale=0.05):
        small_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small_image, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    # Bộ lọc chuyển đổi màu sắc
    def color_transform(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Bộ lọc Đen Trắng
    def black_and_white_filter(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bộ lọc Nhấn Mạnh (Sharpening)
    def sharpen_filter(image):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    # Bộ lọc Hình Bóng (Vignette)
    def vignette_filter(image):
        rows, cols = image.shape[:2]
        X_resultant_kernel = cv2.getGaussianKernel(cols, 200)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, 200)
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        vignette_image = np.copy(image)
        for i in range(3):  # Đối với từng kênh màu
            vignette_image[..., i] = vignette_image[..., i] * mask
        return vignette_image

    # Bộ lọc Lật (Flip)
    def flip_filter(image, direction='horizontal'):
        if direction == 'horizontal':
            return cv2.flip(image, 1)
        else:  # chiều dọc
            return cv2.flip(image, 0)

    # Bộ lọc Nhòe Đường Viền (Edge Blur)
    def edge_blur_filter(image):
        edges = cv2.Canny(image, 100, 200)
        blurred = cv2.GaussianBlur(edges, (5, 5), 0)
        return blurred

    # Áp dụng các bộ lọc
    sketches = sketch_filter(image)
    oil_paintings = oil_paint_filter(image)
    blurred = blur_filter(image)
    mosaics = mosaic_filter(image)
    color_transformed = color_transform(image)
    black_and_white = black_and_white_filter(image)
    sharpened = sharpen_filter(image)
    vignette = vignette_filter(image)
    flipped = flip_filter(image, direction='horizontal')
    edge_blurred = edge_blur_filter(image)

    return (sketches, oil_paintings, blurred, mosaics, color_transformed,
            black_and_white, sharpened, vignette, flipped, edge_blurred)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        clear_output_folder()  # Xóa các tệp trong thư mục output
        file = request.files['image']
        if file:
            filename = file.filename
            image_path = os.path.join(OUTPUT_FOLDER, filename)
            file.save(image_path)

            # Đọc ảnh và áp dụng các bộ lọc
            image = cv2.imread(image_path)
            results = apply_filters(image)

            # Lưu các ảnh đã xử lý
            filter_names = ['sketch', 'oil', 'blurred', 'mosaic', 'color',
                            'bw', 'sharpened', 'vignette', 'flipped', 'edge_blurred']
            for result, name in zip(results, filter_names):
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'{name}_{filename}'), result)

            return redirect(url_for('result', filename=filename))
    
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
