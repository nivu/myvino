python ./face_recognition_demo.py ^
-m_fd E:/openvino/open_model_zoo/tools/downloader/intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml ^
-m_lm E:/openvino/open_model_zoo/tools/downloader/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml ^
-m_reid E:/openvino/open_model_zoo/tools/downloader/intel/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml ^
--verbose ^
-fg "E:/openvino/face_gallery" -d_fd CPU -d_lm CPU -d_reid CPU --allow_grow


./downloader.py --name face-detection-retail-0004 --precisions FP16,INT8

./downloader.py --name landmarks-regression-retail-0009 --precisions FP16,INT8

./downloader.py --name face-reidentification-retail-0095 --precisions FP16,INT8


python downloader.py --name face-detection-retail-0004

python downloader.py --name landmarks-regression-retail-0009

python downloader.py --name face-reidentification-retail-0095



