import pandas as pd
from pathlib import Path

def generate_fraud_review_report(fraud_csv="output/reports/fraud_detection_report.csv",
                                  voter_csv="data/processed/voter_data_complete.csv"):
    """
    Generate a human-friendly review report where humans can decide:
    - Which card to keep as ORIGINAL
    - Which card to reject as FAKE
    """

    # Load data
    frauds_df = pd.read_csv(fraud_csv)
    voters_df = pd.read_csv(voter_csv)

    if len(frauds_df) == 0:
        print("\n✅ NO FRAUDS TO REVIEW - All voters are clean!\n")
        return

    output_dir = Path("output/reviews")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("GENERATING HUMAN REVIEW REPORT")
    print("="*70 + "\n")

    # Create review report
    review_data = []

    for idx, fraud in frauds_df.iterrows():
        card_1 = fraud['card_1']
        card_2 = fraud['card_2']
        fraud_type = fraud['fraud_type']

        # Get full voter data for both cards
        voter_1 = voters_df[voters_df['card_id'] == card_1].iloc[0]
        voter_2 = voters_df[voters_df['card_id'] == card_2].iloc[0]

        if fraud_type == 'FAKE_DETAILS':
            # Both have same details - human needs to decide based on document quality
            review_data.append({
                'fraud_number': idx + 1,
                'fraud_type': 'DUPLICATE_DETAILS',
                'card_1': card_1,
                'card_1_name': voter_1['name'],
                'card_1_father': voter_1['father_husband_name'],
                'card_1_age': voter_1['age'],
                'card_1_house': voter_1['house_number'],
                'card_2': card_2,
                'card_2_name': voter_2['name'],
                'card_2_father': voter_2['father_husband_name'],
                'card_2_age': voter_2['age'],
                'card_2_house': voter_2['house_number'],
                'similarity': '100% (all details match)',
                'recommendation': 'REVIEW DOCUMENT QUALITY - Keep earlier card as original',
                'decision': 'PENDING (Needs human review)'
            })

        elif fraud_type == 'FAKE_FACE':
            # Same person, different details - face match is strong evidence
            similarity = fraud['face_similarity_percent']

            review_data.append({
                'fraud_number': idx + 1,
                'fraud_type': 'DUPLICATE_FACE',
                'card_1': card_1,
                'card_1_name': voter_1['name'],
                'card_1_father': voter_1['father_husband_name'],
                'card_1_age': voter_1['age'],
                'card_1_house': voter_1['house_number'],
                'card_2': card_2,
                'card_2_name': voter_2['name'],
                'card_2_father': voter_2['father_husband_name'],
                'card_2_age': voter_2['age'],
                'card_2_house': voter_2['house_number'],
                'similarity': f'{similarity}% (SAME PERSON)',
                'recommendation': 'LIKELY FRAUD - Same person, different details',
                'decision': 'PENDING (Needs human review)'
            })

    review_df = pd.DataFrame(review_data)

    # Save review report
    review_file = output_dir / "fraud_review_report.csv"
    review_df.to_csv(review_file, index=False, encoding='utf-8-sig')

    print(f"✓ Review report saved: {review_file}\n")

    # Generate human-readable HTML report
    html_content = generate_html_review(review_df, voters_df)
    html_file = output_dir / "fraud_review_report.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✓ HTML review saved: {html_file}\n")

    # Print summary
    print("="*70)
    print("FRAUD REVIEW SUMMARY")
    print("="*70)
    print(f"\nTotal Fraud Suspects: {len(review_df)}")

    duplicate_details = len(review_df[review_df['fraud_type'] == 'DUPLICATE_DETAILS'])
    duplicate_face = len(review_df[review_df['fraud_type'] == 'DUPLICATE_FACE'])

    print(f"  - Same Details: {duplicate_details}")
    print(f"  - Same Face: {duplicate_face}")

    print(f"\n{'─'*70}")
    print("ACTION REQUIRED:")
    print(f"  1. Open: {html_file}")
    print(f"  2. Review each suspect pair")
    print(f"  3. Decide which card to KEEP as original")
    print(f"  4. Reject the other card as FAKE")
    print(f"\n{'='*70}\n")


def generate_html_review(review_df, voters_df):
    """Generate HTML report for easy human review"""

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voter Fraud Review Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .fraud-card {
                border: 2px solid #e74c3c;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                background-color: #fdeaea;
            }
            .card-pair {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 15px 0;
            }
            .voter-card {
                border: 1px solid #bdc3c7;
                padding: 15px;
                border-radius: 5px;
                background-color: #ecf0f1;
            }
            .voter-card h3 { margin-top: 0; }
            .field { margin: 8px 0; }
            .label { font-weight: bold; color: #2c3e50; }
            .value { color: #34495e; }
            .duplicate-details { border-color: #f39c12; background-color: #fef5e7; }
            .duplicate-face { border-color: #e74c3c; background-color: #fadbd8; }
            .recommendation {
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                padding: 10px;
                border-radius: 5px;
                margin-top: 15px;
            }
            .action-needed {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <h1>🚨 Voter Fraud Review Report</h1>
        <p>Review each suspected fraud pair and decide which card to KEEP as the original voter.</p>
    """

    for idx, fraud in review_df.iterrows():
        card_1_data = fraud['card_1_name']
        card_2_data = fraud['card_2_name']

        if fraud['fraud_type'] == 'DUPLICATE_DETAILS':
            title = "⚠️ SAME DETAILS DETECTED"
            class_name = "duplicate-details"
            description = f"Both cards have identical details: {fraud['card_1']} and {fraud['card_2']}"
        else:
            title = "🚨 SAME PERSON DETECTED"
            class_name = "duplicate-face"
            description = f"Same person ({fraud['similarity']}): {fraud['card_1']} and {fraud['card_2']}"

        html += f"""
        <div class="fraud-card {class_name}">
            <h2>{title}</h2>
            <p><strong>Description:</strong> {description}</p>

            <div class="card-pair">
                <div class="voter-card">
                    <h3>Card 1: {fraud['card_1']}</h3>
                    <div class="field">
                        <span class="label">Name:</span>
                        <span class="value">{fraud['card_1_name']}</span>
                    </div>
                    <div class="field">
                        <span class="label">Father/Husband:</span>
                        <span class="value">{fraud['card_1_father']}</span>
                    </div>
                    <div class="field">
                        <span class="label">Age:</span>
                        <span class="value">{fraud['card_1_age']}</span>
                    </div>
                    <div class="field">
                        <span class="label">House:</span>
                        <span class="value">{fraud['card_1_house']}</span>
                    </div>
                </div>

                <div class="voter-card">
                    <h3>Card 2: {fraud['card_2']}</h3>
                    <div class="field">
                        <span class="label">Name:</span>
                        <span class="value">{fraud['card_2_name']}</span>
                    </div>
                    <div class="field">
                        <span class="label">Father/Husband:</span>
                        <span class="value">{fraud['card_2_father']}</span>
                    </div>
                    <div class="field">
                        <span class="label">Age:</span>
                        <span class="value">{fraud['card_2_age']}</span>
                    </div>
                    <div class="field">
                        <span class="label">House:</span>
                        <span class="value">{fraud['card_2_house']}</span>
                    </div>
                </div>
            </div>

            <div class="recommendation">
                <strong>Recommendation:</strong> {fraud['recommendation']}<br>
                <strong>Status:</strong> {fraud['decision']}
            </div>

            <div class="action-needed">
                <strong>👉 YOUR DECISION:</strong><br>
                Which card should be kept as ORIGINAL?
                <br>
                [ ] Card 1: {fraud['card_1']} - KEEP<br>
                [ ] Card 2: {fraud['card_2']} - KEEP<br>
                <br>
                The other card will be marked as FAKE.
            </div>
        </div>
        """

    html += """
        <div class="action-needed" style="background-color: #d4edda; border-color: #c3e6cb;">
            <h2>✅ Next Steps</h2>
            <ol>
                <li>Review each fraud pair above</li>
                <li>Decide which card is ORIGINAL and which is FAKE</li>
                <li>Mark your decisions in the checkboxes</li>
                <li>Save your decisions to: <code>output/reviews/fraud_decisions.csv</code></li>
                <li>System will process rejections and generate final voter list</li>
            </ol>
        </div>
    </body>
    </html>
    """

    return html


if __name__ == "__main__":
    generate_fraud_review_report()
