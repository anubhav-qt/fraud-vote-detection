import fitz
from PIL import Image
import os
from pathlib import Path

class PDFProcessor:
    def __init__(self, pdf_path, output_dir="data/extracted_photos"):
        """
        Initialize PDF processor

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted photos
        """
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Open PDF
        self.doc = fitz.open(pdf_path)
        print(f"Loaded PDF with {len(self.doc)} pages")

    def extract_images_from_page(self, page_num):
        """
        Extract all images from a specific page

        Args:
            page_num: Page number (0-indexed)

        Returns:
            List of extracted image info
        """
        page = self.doc[page_num]
        images = []

        # Get all images on the page
        image_list = page.get_images(full=True)

        print(f"Page {page_num + 1}: Found {len(image_list)} images")

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]  # Image reference number

            # Extract image data
            base_image = self.doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Save image
            image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
            image_path = self.output_dir / image_filename

            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            images.append({
                'page': page_num + 1,
                'index': img_index + 1,
                'path': str(image_path),
                'xref': xref
            })

        return images

    def extract_all_voter_images(self, start_page=2):
        """
        Extract images from all voter pages (starting from page 3 = index 2)

        Args:
            start_page: Page index to start from (default 2 = page 3)

        Returns:
            Dictionary mapping page numbers to extracted images
        """
        all_images = {}

        # Process from start_page to end of document
        for page_num in range(start_page, len(self.doc)):
            images = self.extract_images_from_page(page_num)
            if images:
                all_images[page_num + 1] = images

        print(f"\nTotal images extracted: {sum(len(imgs) for imgs in all_images.values())}")
        return all_images

    def get_page_text(self, page_num):
        """
        Extract raw text from a page for OCR processing

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Page text content
        """
        page = self.doc[page_num]
        return page.get_text()

    def close(self):
        """Close the PDF document"""
        self.doc.close()


# Test the processor
if __name__ == "__main__":
    # Path to your PDF
    pdf_path = "data/input_pdfs/2023-EROLLGEN-S20-55-FinalRoll-Revision1-   HIN-1.pdf"

    # Create processor
    processor = PDFProcessor(pdf_path)

    # Extract images from first voter page (page 3)
    print("Testing extraction on page 3...")
    test_images = processor.extract_images_from_page(2)  # Page 3 = index 2

    print(f"\nExtracted {len(test_images)} images from page 3")
    for img in test_images:
        print(f"  - {img['path']}")

    # Extract text from page 3 to see structure
    print("\n\nSample text from page 3:")
    text = processor.get_page_text(2)
    print(text[:500])  # Print first 500 characters

    processor.close()
