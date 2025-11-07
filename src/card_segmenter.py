import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path

class VoterCardSegmenter:
    def __init__(self, pdf_path, output_dir="data/extracted_cards"):
        """
        Initialize card segmenter

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save individual cards
        """
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.doc = fitz.open(pdf_path)
        print(f"Loaded PDF with {len(self.doc)} pages")

    def pdf_page_to_image(self, page_num, zoom=3):
        """
        Convert PDF page to high-resolution image

        Args:
            page_num: Page number (0-indexed)
            zoom: Zoom factor for better resolution

        Returns:
            OpenCV image (numpy array)
        """
        page = self.doc[page_num]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert to numpy array (OpenCV format)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        # Convert RGB to BGR for OpenCV
        if pix.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def detect_grid_lines(self, image):
        """
        Detect horizontal and vertical grid lines

        Args:
            image: OpenCV image (BGR)

        Returns:
            Tuple of (vertical_lines, horizontal_lines)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply binary threshold (inverted because lines are dark)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Get image dimensions
        height, width = binary.shape

        # Create kernels for line detection
        # Vertical lines: tall and thin
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 30))

        # Horizontal lines: wide and short
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 30, 1))

        # Detect vertical lines
        vertical_lines = cv2.erode(binary, vertical_kernel, iterations=2)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=2)

        # Detect horizontal lines
        horizontal_lines = cv2.erode(binary, horizontal_kernel, iterations=2)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)

        return vertical_lines, horizontal_lines

    def find_line_positions(self, line_image, direction='vertical'):
        """
        Find positions of lines in the image

        Args:
            line_image: Binary image containing lines
            direction: 'vertical' or 'horizontal'

        Returns:
            List of line positions (x for vertical, y for horizontal)
        """
        if direction == 'vertical':
            # Sum along columns to find vertical lines
            projection = np.sum(line_image, axis=0)
        else:
            # Sum along rows to find horizontal lines
            projection = np.sum(line_image, axis=1)

        # Find peaks in projection (where lines are)
        threshold = np.max(projection) * 0.3  # 30% of max
        line_positions = []

        for i in range(len(projection)):
            if projection[i] > threshold:
                # Check if this is a new line (not part of previous thick line)
                if not line_positions or i - line_positions[-1] > 20:
                    line_positions.append(i)

        return line_positions

    def extract_cards_from_grid(self, image, v_lines, h_lines):
        """
        Extract individual cards based on grid lines

        Args:
            image: Original image
            v_lines: List of vertical line positions
            h_lines: List of horizontal line positions

        Returns:
            List of card bounding boxes [(x, y, w, h), ...]
        """
        cards = []

        # Create cells from grid intersections
        for i in range(len(v_lines) - 1):
            for j in range(len(h_lines) - 1):
                x1 = v_lines[i]
                x2 = v_lines[i + 1]
                y1 = h_lines[j]
                y2 = h_lines[j + 1]

                # Calculate dimensions
                w = x2 - x1
                h = y2 - y1

                # Filter out very small or very large boxes
                # Cards should be reasonable size
                if w > 100 and h > 100 and w < image.shape[1] * 0.9 and h < image.shape[0] * 0.9:
                    cards.append((x1, y1, w, h))

        return cards

    def crop_cards_from_image(self, image, card_boxes, page_num):
        """
        Crop individual cards from page image

        Args:
            image: OpenCV image
            card_boxes: List of bounding boxes
            page_num: Page number for naming

        Returns:
            List of dictionaries with card info
        """
        cards = []

        for idx, (x, y, w, h) in enumerate(card_boxes):
            # Add small padding and ensure within bounds
            padding = 5
            x1 = max(0, x + padding)
            y1 = max(0, y + padding)
            x2 = min(image.shape[1], x + w - padding)
            y2 = min(image.shape[0], y + h - padding)

            # Crop card
            card_img = image[y1:y2, x1:x2]

            # Skip if card is too small
            if card_img.shape[0] < 50 or card_img.shape[1] < 50:
                continue

            # Save card image
            card_filename = f"page_{page_num + 1}_card_{idx + 1}.jpg"
            card_path = self.output_dir / card_filename
            cv2.imwrite(str(card_path), card_img)

            cards.append({
                'page': page_num + 1,
                'card_index': idx + 1,
                'path': str(card_path),
                'bbox': (x, y, w, h),
                'dimensions': (w, h)
            })

        return cards

    def visualize_detections(self, image, card_boxes, v_lines, h_lines, page_num):
        """
        Draw grid lines and bounding boxes on image

        Args:
            image: OpenCV image
            card_boxes: List of bounding boxes
            v_lines: Vertical line positions
            h_lines: Horizontal line positions
            page_num: Page number for naming
        """
        vis_img = image.copy()

        # Draw vertical lines in blue
        for x in v_lines:
            cv2.line(vis_img, (x, 0), (x, image.shape[0]), (255, 0, 0), 2)

        # Draw horizontal lines in green
        for y in h_lines:
            cv2.line(vis_img, (0, y), (image.shape[1], y), (0, 255, 0), 2)

        # Draw card boxes in red
        for idx, (x, y, w, h) in enumerate(card_boxes):
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

            # Add label
            label = f"{idx + 1}"
            cv2.putText(vis_img, label, (x + 10, y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Save visualization
        vis_path = self.output_dir / f"page_{page_num + 1}_visualization.jpg"
        cv2.imwrite(str(vis_path), vis_img)
        print(f"  Visualization saved: {vis_path.name}")

    def process_page(self, page_num, visualize=True):
        """
        Process a single page to extract all cards

        Args:
            page_num: Page number (0-indexed)
            visualize: Whether to save visualization

        Returns:
            List of card info dictionaries
        """
        print(f"\nProcessing page {page_num + 1}...")

        # Convert PDF page to image
        image = self.pdf_page_to_image(page_num)
        print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

        # Detect grid lines
        vertical_lines, horizontal_lines = self.detect_grid_lines(image)

        # Find line positions
        v_line_positions = self.find_line_positions(vertical_lines, 'vertical')
        h_line_positions = self.find_line_positions(horizontal_lines, 'horizontal')

        print(f"  Found {len(v_line_positions)} vertical lines, {len(h_line_positions)} horizontal lines")

        if len(v_line_positions) < 2 or len(h_line_positions) < 2:
            print("  ⚠ Insufficient grid lines detected")
            return []

        # Extract cards from grid
        card_boxes = self.extract_cards_from_grid(image, v_line_positions, h_line_positions)
        print(f"  Detected {len(card_boxes)} cards")

        if visualize and card_boxes:
            self.visualize_detections(image, card_boxes, v_line_positions, h_line_positions, page_num)

        # Crop and save cards
        cards = self.crop_cards_from_image(image, card_boxes, page_num)
        print(f"  ✓ Saved {len(cards)} cards")

        return cards

    def process_all_pages(self, start_page=2, end_page=None, sample_size=None):
        """
        Process multiple pages

        Args:
            start_page: Starting page (0-indexed)
            end_page: Ending page (None = all)
            sample_size: Number of pages to process (None = all)

        Returns:
            List of all card info
        """
        if end_page is None:
            end_page = len(self.doc)

        if sample_size:
            end_page = min(start_page + sample_size, end_page)

        all_cards = []

        for page_num in range(start_page, end_page):
            cards = self.process_page(page_num)
            all_cards.extend(cards)

        print(f"\n{'='*60}")
        print(f"✓ Total cards extracted: {len(all_cards)}")
        if all_cards:
            print(f"  From {end_page - start_page} pages")
            print(f"  Average: {len(all_cards) / (end_page - start_page):.1f} cards/page")
        print(f"{'='*60}")

        return all_cards

    def close(self):
        """Close PDF document"""
        self.doc.close()


# Test
if __name__ == "__main__":
    pdf_path = "data/input_pdfs/2023-EROLLGEN-S20-55-FinalRoll-Revision1-   HIN-1.pdf"

    segmenter = VoterCardSegmenter(pdf_path)

    # Test on pages 3-7
    print("Testing grid-based card segmentation...")
    cards = segmenter.process_all_pages(start_page=2, sample_size=5)

    if cards:
        print(f"\nFirst 10 extracted cards:")
        for card in cards[:10]:
            print(f"  {card['path']}")

    segmenter.close()
