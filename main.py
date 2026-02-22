import os
# Suppress PaddleOCR logging and fix PIR/oneDNN executor issues on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PADDLE_USE_MKLDNN'] = '0'
os.environ['PADDLE_DISABLE_DNNL'] = '1'
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_enable_pir_api'] = '0' 
os.environ['FLAGS_enable_pir_in_executor'] = '0'
os.environ['FLAGS_enable_new_executor'] = '0'
os.environ['FLAGS_enable_new_ir'] = '0'
os.environ['FLAGS_enable_onednn'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_cuda'] = '0'



import sys
import argparse
import logging
from vin_extractor.pipeline import VINExtractionPipeline

# Configure minimal logging for production
logging.basicConfig(level=logging.ERROR, format='%(message)s')

def main():
    parser = argparse.ArgumentParser(description="Industrial VIN Extraction System")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR if available")
    
    args = parser.parse_args()

    try:
        pipeline = VINExtractionPipeline(use_gpu=args.gpu)
        result = pipeline.process_image(args.image_path)
        print(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
        print("Invalid VIN")

if __name__ == "__main__":
    main()
