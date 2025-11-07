import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from card_segmenter import VoterCardSegmenter
from google_vision import GoogleVisionExtractor
from duplicate_detector import DuplicateDetectorFinal

def main():
    """
    Complete pipeline:
    1. Extract cards from all PDFs
    2. Extract text and faces using Google Vision
    3. Detect frauds
    """

    print("\n" + "="*70)
    print("VOTER FRAUD DETECTION SYSTEM - COMPLETE PIPELINE")
    print("="*70)

    input_pdfs_dir = Path("data/input_pdfs")

    # Find all PDFs
    pdf_files = list(input_pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"\n❌ No PDF files found in {input_pdfs_dir}")
        return

    print(f"\n✓ Found {len(pdf_files)} PDF file(s)\n")

    # =========================================================================
    # STEP 1: EXTRACT CARDS FROM ALL PDFS
    # =========================================================================
    print("="*70)
    print("STEP 1: EXTRACTING VOTER CARDS FROM PDFS")
    print("="*70 + "\n")

    total_cards = 0
    for pdf_file in pdf_files:
        print(f"📄 Processing: {pdf_file.name}\n")
        segmenter = VoterCardSegmenter(str(pdf_file))

        # Process all pages starting from page 3 (index 2)
        cards = segmenter.process_all_pages(start_page=2, end_page=None)
        total_cards += len(cards)

        segmenter.close()

    print(f"\n{'─'*70}")
    print(f"✓ Total cards extracted: {total_cards}\n")

    # =========================================================================
    # STEP 2: EXTRACT TEXT AND FACES USING GOOGLE VISION
    # =========================================================================
    print("="*70)
    print("STEP 2: EXTRACTING TEXT AND FACES")
    print("="*70 + "\n")

    extractor = GoogleVisionExtractor()
    voter_data = extractor.process_all_cards()  # Process ALL cards

    # Save voter data
    voter_csv = Path("data/processed/voter_data_complete.csv")
    voter_data.to_csv(voter_csv, index=False, encoding='utf-8-sig')

    print(f"\n{'─'*70}")
    print(f"✓ Voter data saved: {voter_csv}\n")

    # =========================================================================
    # STEP 3: DETECT FRAUDS
    # =========================================================================
    print("="*70)
    print("STEP 3: DETECTING VOTER FRAUDS")
    print("="*70 + "\n")

    detector = DuplicateDetectorFinal(str(voter_csv))
    frauds = detector.detect_all_frauds()

    # =========================================================================
    # STEP 4: GENERATE FINAL REPORT
    # =========================================================================
    print("="*70)
    print("STEP 4: GENERATING FINAL REPORT")
    print("="*70 + "\n")

    detector.generate_report(frauds)

    # Print summary
    print("="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"\nTotal Voters Processed: {len(voter_data)}")
    print(f"Total Frauds Detected: {len(frauds)}")

    if len(frauds) > 0:
        scenario_1 = len(frauds[frauds['fraud_type'] == 'FAKE_DETAILS'])
        scenario_2 = len(frauds[frauds['fraud_type'] == 'FAKE_FACE'])
        print(f"  - Fake Details: {scenario_1}")
        print(f"  - Fake Face: {scenario_2}")
        fraud_rate = (len(frauds) / len(voter_data)) * 100
        print(f"  - Fraud Rate: {fraud_rate:.2f}%")
    else:
        print("  ✅ No frauds detected!")

    print(f"\n{'='*70}\n")
    print("✓ PIPELINE COMPLETE!\n")
    print("Output files saved to:")
    print(f"  - Voter data: {voter_csv}")
    print(f"  - Fraud report: output/reports/fraud_detection_report.csv")
    print(f"  - Face images: data/processed/photos/\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user\n")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
