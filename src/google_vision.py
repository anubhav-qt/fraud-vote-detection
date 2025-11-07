import cv2
import numpy as np
import face_recognition
from pathlib import Path
import pandas as pd
import re
from google.cloud import vision

class GoogleVisionExtractor:
    def __init__(self, cards_dir="data/extracted_cards", output_dir="data/processed"):
        """Extract voter data using Google Cloud Vision API (98%+ accuracy)"""
        self.cards_dir = Path(cards_dir)
        self.output_dir = Path(output_dir)
        self.photos_dir = self.output_dir / "photos"
        self.photos_dir.mkdir(parents=True, exist_ok=True)

        self.client = vision.ImageAnnotatorClient()
        print("✓ Google Vision API ready\n")

    def extract_face(self, card_image):
        """Extract face with multiple detection methods"""
        rgb_image = cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB)

        # Method 1: Try HOG (fast)
        face_locations = face_recognition.face_locations(rgb_image, model='hog')

        # Method 2: If HOG fails, try CNN (slower but more accurate)
        if not face_locations:
            try:
                face_locations = face_recognition.face_locations(rgb_image, model='cnn')
            except:
                pass

        # Method 3: If face_recognition fails, try OpenCV Haar Cascade (fallback)
        if not face_locations:
            gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                # Convert Haar format [x, y, w, h] to face_recognition format [top, right, bottom, left]
                x, y, w, h = faces[0]
                face_locations = [(y, x+w, y+h, x)]

        if not face_locations:
            return None, None

        # Get face encoding
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if not face_encodings:
            return None, None

        # Extract and save face
        top, right, bottom, left = face_locations[0]
        padding = 10
        face_image = card_image[
            max(0, top-padding):min(card_image.shape[0], bottom+padding),
            max(0, left-padding):min(card_image.shape[1], right+padding)
        ]

        return face_image, face_encodings[0]


    def extract_text_region(self, card_image):
        """Extract left 60%"""
        height, width = card_image.shape[:2]
        return card_image[:, :int(width * 0.6)]

    def ocr_with_google_vision(self, image):
        """Extract text using Google Cloud Vision API"""
        try:
            # Convert to bytes
            success, encoded_image = cv2.imencode('.jpg', image)
            image_content = encoded_image.tobytes()

            # Create Vision image
            image_obj = vision.Image(content=image_content)

            # Perform document text detection (best for structured docs)
            response = self.client.document_text_detection(
                image=image_obj,
                image_context=vision.ImageContext(language_hints=['hi', 'en'])
            )

            text = response.full_text_annotation.text

            print(f"    → Google Vision extracted:")
            for i, line in enumerate(text.split('\n')[:10]):
                if line.strip():
                    print(f"       {i+1}. '{line}'")

            return text

        except Exception as e:
            print(f"    → Google Vision error: {e}")
            return ""

    def parse_structured_fields(self, text):
        """Parse all 5 fields with flexible matching"""
        details = {
            'name': None,
            'father_husband_name': None,
            'house_number': None,
            'age': None,
            'gender': None
        }

        lines = [line.strip() for line in text.split('\n') if line.strip()]

        for i, line in enumerate(lines):
            # Name: निर्वाचक का नाम
            if re.search(r'निर्[वब]ाचक.*नाम', line, re.IGNORECASE):
                match = re.search(r'नाम\s*[:;!]\s*(.+)', line)
                if match:
                    name = match.group(1).strip()
                    if len(name) > 2 and not any(x in name for x in ['पिता', 'पति', 'मकान', 'उम्र']):
                        details['name'] = name
                elif i + 1 < len(lines) and not re.search(r'पिता|पति|मकान|उम्र', lines[i+1]):
                    details['name'] = lines[i+1].strip()

            # Father/Husband: पिता/पति का नाम
            if re.search(r'(पिता|प्रति|पत्ति|पति).*नाम', line, re.IGNORECASE):
                match = re.search(r'नाम\s*[:;!]\s*(.+)', line)
                if match:
                    parent = match.group(1).strip()
                    if len(parent) > 2 and 'मकान' not in parent:
                        details['father_husband_name'] = parent
                elif i + 1 < len(lines) and not re.search(r'मकान|उम्र', lines[i+1]):
                    details['father_husband_name'] = lines[i+1].strip()

            # House: मकान संख्या
            if re.search(r'(मकान|TA|ग्रकान).*(संख्या|मंख्या)', line, re.IGNORECASE):
                match = re.search(r'[:;!*]\s*([०-९0-9]+)', line)
                if match:
                    details['house_number'] = match.group(1)
                elif i + 1 < len(lines):
                    num_match = re.search(r'([०-९0-9]+)', lines[i+1])
                    if num_match:
                        details['house_number'] = num_match.group(1)

            # Age and Gender: उम्र: age लिंग: gender
            if re.search(r'(उम्र|उप्र)', line, re.IGNORECASE):
                # Extract age
                age_match = re.search(r'[:;]\s*([०-९0-9]{2,3})', line)
                if age_match:
                    age_str = age_match.group(1)
                    # Convert Devanagari to Arabic
                    age_str = (age_str.replace('०', '0').replace('१', '1').replace('२', '2')
                              .replace('३', '3').replace('४', '4').replace('५', '5')
                              .replace('६', '6').replace('७', '7').replace('८', '8').replace('९', '9'))
                    age = int(age_str)
                    if 18 <= age <= 120:
                        details['age'] = age

                # Extract gender
                if re.search(r'लिंग', line):
                    if re.search(r'महिला|लिंग\s*[:;]\s*म', line):
                        details['gender'] = 'F'
                    elif re.search(r'पुरुष|लिंग\s*[:;]\s*पु', line):
                        details['gender'] = 'M'

        return details

    def process_card(self, card_path):
        """Process single card"""
        card_path = Path(card_path)
        card_id = card_path.stem

        card_img = cv2.imread(str(card_path))
        if card_img is None:
            return None

        # Extract face
        face_img, face_encoding = self.extract_face(card_img)

        # Save face
        face_path = None
        if face_img is not None:
            face_filename = f"{card_id}_face.jpg"
            face_path = self.photos_dir / face_filename
            cv2.imwrite(str(face_path), face_img)

        # Extract text region
        text_region = self.extract_text_region(card_img)

        # OCR with Google Vision
        text = self.ocr_with_google_vision(text_region)

        # Parse
        details = self.parse_structured_fields(text)

        return {
            'card_id': card_id,
            'face_path': str(face_path) if face_path else None,
            'face_encoding': face_encoding.tolist() if face_encoding is not None else None,
            **details
        }

    def process_all_cards(self, limit=None):
        """Process all cards"""
        card_files = sorted(self.cards_dir.glob("page_*_card_*.jpg"))

        if limit:
            card_files = card_files[:limit]

        print(f"Processing {len(card_files)} cards with Google Vision...\n")

        results = []
        for i, card_path in enumerate(card_files, 1):
            result = self.process_card(card_path)

            if result:
                results.append(result)
                print(f"[{i}/{len(card_files)}] {card_path.name}")
                print(f"  Name: {result['name']}")
                print(f"  Father/Husband: {result['father_husband_name']}")
                print(f"  House: {result['house_number']}")
                print(f"  Age: {result['age']}")
                print(f"  Gender: {result['gender']}")
                print(f"  Face: {'✓' if result['face_encoding'] else '✗'}\n")

        df = pd.DataFrame(results)

        print(f"{'='*60}")
        print(f"✓ Processed: {len(df)} cards")
        print(f"  Names: {df['name'].notna().sum()}")
        print(f"  Father/Husband: {df['father_husband_name'].notna().sum()}")
        print(f"  House: {df['house_number'].notna().sum()}")
        print(f"  Age: {df['age'].notna().sum()}")
        print(f"  Gender: {df['gender'].notna().sum()}")
        print(f"  Faces: {df['face_encoding'].notna().sum()}")
        print(f"{'='*60}")

        return df


if __name__ == "__main__":
    extractor = GoogleVisionExtractor()
    df = extractor.process_all_cards(limit=10)

    output_file = extractor.output_dir / "voter_data_google_vision.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Saved to {output_file}")
