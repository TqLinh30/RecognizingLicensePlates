# RecognizingLicensePlates

Hệ thống nhận dạng biển số xe được xây dựng từ đầu bằng Python. Dự án chủ
đích không dùng OpenCV, scikit-image hay framework computer vision cấp cao.
Toàn bộ thuật toán xử lý ảnh, phân đoạn ký tự, trích xuất đặc trưng và nhận
dạng ký tự được cài đặt thủ công bằng NumPy; Pillow chỉ dùng để đọc/ghi ảnh.

Mục tiêu chính của dự án là học và kiểm soát rõ từng bước trong pipeline ALPR
(Automatic License Plate Recognition), thay vì gọi sẵn một thư viện lớn.

## Trạng thái hiện tại

- Pipeline hoàn chỉnh từ ảnh đầu vào đến chuỗi ký tự OCR.
- Có giao diện desktop để chọn ảnh trực tiếp từ máy tính.
- Có chế độ hiển thị kết quả trung gian của từng bước nhận dạng.
- Có model OCR đi kèm trong `data/models`.
- Có bộ ảnh mẫu trong `data/samples` và file nhãn trong
  `data/labels/sample_ocr_labels.json`.
- Benchmark local hiện tại: `24/24` ảnh mẫu pass bằng
  `python -m scripts.evaluate_samples`.
- Unit tests hiện tại: `95 passed` bằng `python -m pytest -q`.

## Các bước xử lý đã xây dựng

Pipeline tổng quát:

```text
Input image
  -> Step 1: Preprocessing
  -> Step 2: Plate detection
  -> Step 3: Plate cropping and normalization
  -> Step 4: Character segmentation
  -> Step 5: Feature extraction
  -> Step 6: Character classification
  -> Step 7: OCR result
```

Chi tiết:

1. Preprocessing
   - Chuyển RGB sang grayscale theo công thức luminance.
   - Gaussian blur tự cài đặt bằng convolution tách 1D.
   - Median filter.
   - Histogram equalization.
   - CLAHE tự cài đặt để tăng tương phản cục bộ.
   - Otsu thresholding tự cài đặt.

2. Plate detection
   - Sobel-X/Sobel-Y tự cài đặt.
   - Morphology: dilation, erosion, opening, closing.
   - Connected components bằng thuật toán two-pass và Union-Find.
   - Lọc candidate theo tỉ lệ khung, diện tích, fill ratio và mật độ gradient.
   - Có fallback cho ảnh xe lớn có biển sáng nhưng detector cạnh bỏ sót.

3. Plate normalization
   - Crop vùng biển số.
   - Ước lượng góc nghiêng bằng Hough transform.
   - Xoay ảnh bằng affine transform và bilinear interpolation.
   - Resize biển số về kích thước chuẩn cho bước segment.

4. Character segmentation
   - Threshold ảnh biển số.
   - Làm sạch bằng morphology.
   - Tìm connected components của ký tự.
   - Gom ký tự theo hàng.
   - Chuẩn hóa mỗi ký tự về canvas nhị phân `32x32`.
   - Có projection recovery để tách ký tự bị dính.
   - Có edge-artifact pruning để bỏ logo, viền, city text, badge hoặc mảnh
     trang trí ở rìa mà không cắt nhầm ký tự thật.

5. Feature extraction
   - HOG tự cài đặt: gradient, magnitude, orientation bins, block normalize.
   - Zoning feature: chia ảnh ký tự thành lưới và đo mật độ pixel.
   - Kết hợp feature vector phục vụ classifier.

6. Character classification
   - KNN baseline.
   - MLP tự cài đặt bằng NumPy: forward, ReLU, softmax, backprop.
   - Pixel-template classifier.
   - Zoning-template classifier.
   - Sample-template memory từ các ảnh mẫu thật trong project.
   - GUI hiện blend nhiều model và ưu tiên kết quả ký tự, không ép theo format
     biển số Việt Nam.

7. Output
   - Trả về chuỗi OCR ký tự.
   - Hiển thị confidence trung bình và top candidates trong GUI.
   - Không còn ép kết quả theo format quốc gia cụ thể.

## Cấu trúc thư mục

```text
RecognizingLicensePlates/
|-- src/
|   |-- preprocessing/       Step 1: grayscale, blur, CLAHE, Otsu
|   |-- detection/           Step 2: Sobel, morphology, connected components
|   |-- normalization/       Step 3: crop, Hough, rotate, resize
|   |-- segmentation/        Step 4: character segmentation
|   |-- features/            Step 5: HOG and zoning
|   |-- classifiers/         Step 6: KNN, MLP, template classifiers
|   |-- recognition/         End-to-end pipeline and plate selection
|   |-- datasets/            EMNIST and synthetic character utilities
|   |-- gui/                 Desktop GUI
|   `-- utils/               Image I/O and visualization helpers
|-- scripts/
|   |-- evaluate_samples.py
|   |-- train_emnist_mlp.py
|   |-- train_pixel_template.py
|   |-- train_sample_templates.py
|   |-- train_synthetic_mlp.py
|   `-- train_zoning_template.py
|-- tests/
|-- docs/
|-- data/
|   |-- labels/
|   |-- models/
|   |-- samples/
|   `-- output/
|-- gui.py
|-- requirements.txt
|-- CHANGELOG.md
`-- README.md
```

## Cài đặt

Yêu cầu:

- Python 3.10 trở lên.
- Windows, Linux hoặc macOS.

Clone project:

```bash
git clone https://github.com/TqLinh30/RecognizingLicensePlates.git
cd RecognizingLicensePlates
```

Tạo môi trường ảo:

```bash
python -m venv venv
```

Kích hoạt môi trường ảo trên Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script, chạy:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Cài dependency:

```bash
pip install -r requirements.txt
```

## Chạy bằng Visual Studio Code

1. Mở VS Code.
2. Chọn `File -> Open Folder...`.
3. Mở thư mục `RecognizingLicensePlates`.
4. Mở terminal trong VS Code bằng `Terminal -> New Terminal`.
5. Tạo và kích hoạt virtual environment như phần cài đặt.
6. Chạy GUI:

```bash
python gui.py
```

Hoặc:

```bash
python -m src.gui.app
```

## Sử dụng GUI

GUI cho phép:

- Chọn ảnh từ máy tính bằng file picker.
- Chạy toàn bộ pipeline nhận dạng.
- Xem từng bước trung gian:
  - ảnh grayscale,
  - ảnh sau blur/CLAHE,
  - candidate biển số,
  - biển số sau crop/normalize,
  - ảnh binary trước segment,
  - character boxes,
  - normalized character crops,
  - feature summary,
  - kết quả OCR.

Ảnh đầu vào có thể là:

- ảnh xe đầy đủ có biển số,
- ảnh biển số đã crop,
- ảnh synthetic dùng để debug.

## Benchmark và kiểm thử

Chạy benchmark trên bộ ảnh mẫu:

```bash
python -m scripts.evaluate_samples
```

Kết quả kỳ vọng hiện tại:

```text
24/24 samples pass
```

Chạy unit tests:

```bash
python -m pytest -q
```

Kết quả kỳ vọng hiện tại:

```text
95 passed
```

Kiểm tra cú pháp Python:

```bash
python -m compileall -q src scripts tests gui.py
```

## Dữ liệu và model

Project đi kèm:

- `data/samples`: ảnh mẫu dùng để demo và benchmark.
- `data/labels/sample_ocr_labels.json`: ground truth cho ảnh mẫu.
- `data/models/plate_synthetic_mlp.npz`: MLP train bằng synthetic character data.
- `data/models/plate_pixel_templates.npz`: pixel-template OCR model.
- `data/models/plate_zoning_templates.npz`: zoning-template OCR model.
- `data/models/plate_sample_templates.npz`: template memory tạo từ ảnh mẫu thật.
- `data/models/emnist_mlp.npz`: model thử nghiệm từ EMNIST.

Train lại sample-template model:

```bash
python -m scripts.train_sample_templates
```

Train lại synthetic MLP:

```bash
python -m scripts.train_synthetic_mlp
```

Train lại template models:

```bash
python -m scripts.train_pixel_template
python -m scripts.train_zoning_template
```

## Những khó khăn đã gặp và cách xử lý

1. Ánh sáng không đều
   - Vấn đề: ảnh xe thật có bóng đổ, phản sáng, vùng biển tối/sáng khác nhau.
   - Cách xử lý: thêm CLAHE để tăng tương phản cục bộ trước khi detect và
     threshold.

2. Detector chọn nhầm vùng chữ trên thân xe
   - Vấn đề: các chữ như `TURBO` hoặc chi tiết lưới tản nhiệt cũng có nhiều
     cạnh dọc.
   - Cách xử lý: candidate scoring kết hợp aspect ratio, fill ratio, area và
     gradient density, không chỉ dựa vào cạnh.

3. Ảnh xe lớn nhưng biển số nhỏ
   - Vấn đề: Sobel/morphology có thể bỏ sót biển số sáng nhỏ trong ảnh lớn.
   - Cách xử lý: thêm bright-region fallback và plate selector dựa trên chất
     lượng segmentation downstream.

4. Ký tự bị dính hoặc bị tách sai ở bước 4.2
   - Vấn đề: connected components đôi khi merge nhiều ký tự, hoặc viền/logo bị
     nhận thành ký tự.
   - Cách xử lý: thêm projection-based recovery để tách slot rộng, đồng thời
     thêm edge-artifact pruning để bỏ badge, city text, frame fragments.

5. OCR nhầm giữa các ký tự giống nhau
   - Vấn đề: `0/O`, `1/I`, `5/S`, `7/1`, `8/B` rất dễ nhầm khi ảnh mờ hoặc crop
     bị lệch.
   - Cách xử lý: blend nhiều classifier, thêm template memory từ sample thật,
     cải thiện synthetic training bằng nhiều font, stroke và jitter.

6. Kết quả từng bị ép theo format Việt Nam
   - Vấn đề: ép format giúp một vài biển Việt Nam đúng hơn nhưng làm sai các
     ảnh không theo format đó.
   - Cách xử lý: bỏ Vietnam-format output trong GUI; mục tiêu hiện tại là nhận
     đúng từng ký tự độc lập.

7. Dữ liệu sample không được push lên Git
   - Vấn đề: label/test phụ thuộc vào ảnh mẫu nhưng `data/samples` từng bị
     ignore.
   - Cách xử lý: đưa bộ sample nhỏ vào repo để benchmark có thể chạy lại sau
     khi clone.

## Tự đánh giá khách quan

Điểm mạnh:

- Pipeline rõ ràng, dễ học, dễ debug từng bước.
- Không phụ thuộc OpenCV hoặc thư viện CV cao cấp.
- Có GUI thực tế để chọn ảnh và quan sát từng stage.
- Có test cho nhiều module lõi và benchmark ảnh mẫu.
- Có lịch sử phát triển theo Gitflow, changelog và release tag.

Hạn chế:

- Đây chưa phải hệ thống production ALPR.
- Dataset còn nhỏ, benchmark `24/24` chỉ chứng minh project xử lý tốt bộ sample
  hiện có, không đảm bảo tổng quát trên mọi ảnh ngoài đời.
- Detector cổ điển vẫn yếu trước ảnh quá nghiêng, motion blur, biển quá nhỏ,
  biển bẩn, che khuất hoặc ánh sáng cực đoan.
- OCR dựa nhiều vào template/synthetic data nên có thể nhầm khi font rất khác
  hoặc crop ký tự sai.
- Chưa có perspective transform đầy đủ cho biển bị chụp xiên mạnh.
- Chưa có CI/CD tự động trên GitHub Actions.

Hướng phát triển hợp lý tiếp theo:

- Thu thập thêm dataset thật và chia train/validation/test rõ ràng.
- Thêm annotation cho bounding box biển số và character boxes.
- Train CNN nhỏ bằng NumPy hoặc cho phép optional deep-learning backend.
- Cải thiện perspective correction.
- Thêm GitHub Actions chạy tests và sample benchmark.
- Tách rõ chế độ educational-from-scratch và chế độ production/optional-libs.

## Gitflow và commitflow

Workflow đang dùng:

```text
feature/* -> develop -> release/* -> main + tag -> develop
```

Tài liệu liên quan:

- `docs/gitflow.md`
- `docs/commitflow.md`
- `CHANGELOG.md`

## Ghi chú về dung lượng

Các thư mục sau là artifact local và không nên đưa vào release zip:

- `venv/`
- `.git/`
- `.pytest_cache/`
- `__pycache__/`
- `data/raw/`
- `data/cache/`
- `data/output/`
- `dist/`

Bản zip release được tạo từ source code, docs, tests, model nhỏ và sample ảnh,
giữ dung lượng dưới 10MB.
