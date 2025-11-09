import pandas as pd
import numpy as np
import json
import face_recognition
from pathlib import Path

class DuplicateDetectorFinal:
    def __init__(self, data_csv="data/processed/voter_data_google_vision.csv"):
        """
        Detect fake voters in 2 detectable scenarios:

        SCENARIO 1: FAKE DETAILS
        - Same name + father/husband + (optional) age/gender = FAKE

        SCENARIO 2: FAKE FACE
        - Same face 90%+ similarity = FAKE (same person, different details)

        SCENARIO 3: EVERYTHING FAKE
        - Undetectable (can't do anything about this)
        """
        self.df = pd.read_csv(data_csv)

        # Load face encodings
        self.df['face_encoding'] = self.df['face_encoding'].apply(self._load_encoding)

        print(f"\n{'='*70}")
        print("VOTER FRAUD DETECTION SYSTEM - FINAL VERSION")
        print(f"{'='*70}")
        print(f"\nLoaded {len(self.df)} voter records")
        print(f"  ✓ Faces detected: {self.df['face_encoding'].notna().sum()}")
        print(f"  ✓ Names extracted: {self.df['name'].notna().sum()}")
        print(f"  ✓ Father/Husband names: {self.df['father_husband_name'].notna().sum()}")
        print(f"  ✓ Ages extracted: {self.df['age'].notna().sum()}")
        print(f"  ✓ Genders extracted: {self.df['gender'].notna().sum()}\n")

    def _load_encoding(self, encoding_str):
        """Convert JSON string to numpy array"""
        if pd.isna(encoding_str) or encoding_str == 'None':
            return None
        try:
            return np.array(json.loads(encoding_str))
        except:
            return None

    def detect_scenario_1_fake_details(self):
        """
        SCENARIO 1: FAKE DETAILS DETECTION (Flexible Matching)

        Detection Logic:
        - MUST have: name + father/husband (these are most reliable)
        - IF both names match AND both fathers match:
          - Check age: if both have age, they MUST match (else skip)
          - Check gender: if both have gender, they MUST match (else skip)
          - If all available data matches = FRAUD

        Why this works:
        - Same name + same father/husband = same household
        - Can't be two different people with identical name & father
        - Even if age/gender missing, this combo is strong evidence
        """
        print("="*70)
        print("SCENARIO 1: FAKE DETAILS DETECTION")
        print("="*70)
        print("Rule: Same name + father/husband + (optional age/gender) = FAKE")
        print("Logic: Work with available data, don't penalize incomplete records\n")

        fakes_scenario_1 = []

        for i in range(len(self.df)):
            for j in range(i + 1, len(self.df)):
                row1 = self.df.iloc[i]
                row2 = self.df.iloc[j]

                # REQUIREMENT 1: Both must have name
                if pd.isna(row1['name']) or pd.isna(row2['name']):
                    continue

                # REQUIREMENT 2: Both must have father/husband name
                if pd.isna(row1['father_husband_name']) or pd.isna(row2['father_husband_name']):
                    continue

                # CHECK 1: Names must match exactly
                name_match = row1['name'] == row2['name']
                if not name_match:
                    continue

                # CHECK 2: Father/husband names must match exactly
                father_match = row1['father_husband_name'] == row2['father_husband_name']
                if not father_match:
                    continue

                # At this point: name + father match, now check optional fields
                matching_fields = ["name", "father/husband"]

                # OPTIONAL CHECK 3: If both have age, they must match
                if pd.notna(row1['age']) and pd.notna(row2['age']):
                    if row1['age'] != row2['age']:
                        # Different ages = probably different people, skip
                        continue
                    matching_fields.append("age")

                # OPTIONAL CHECK 4: If both have gender, they must match
                if pd.notna(row1['gender']) and pd.notna(row2['gender']):
                    if row1['gender'] != row2['gender']:
                        # Different gender = definitely different person, skip
                        continue
                    matching_fields.append("gender")

                # If we got here, all available data matches = FRAUD
                fakes_scenario_1.append({
                    'fraud_type': 'FAKE_DETAILS',
                    'card_1': row1['card_id'],
                    'card_2': row2['card_id'],
                    'name': row1['name'],
                    'father_husband': row1['father_husband_name'],
                    'age_1': row1['age'] if pd.notna(row1['age']) else 'N/A',
                    'age_2': row2['age'] if pd.notna(row2['age']) else 'N/A',
                    'gender_1': row1['gender'] if pd.notna(row1['gender']) else 'N/A',
                    'gender_2': row2['gender'] if pd.notna(row2['gender']) else 'N/A',
                    'matching_fields': ' + '.join(matching_fields),
                    'confidence': '100%',
                    'likelihood': 'DEFINITE FRAUD'
                })

        print(f"✓ Found {len(fakes_scenario_1)} fake voters with identical details\n")

        return fakes_scenario_1

    def detect_scenario_2_fake_face(self):
        """
        SCENARIO 2: FAKE FACE DETECTION (With Evidence Scoring)

        Detection Logic:
        - Face similarity >= 90% = SAME PERSON
        - Base confidence: 90% (from face match)
        - Add evidence from details:
          - If names are DIFFERENT: +5% (SUPER SUSPICIOUS - same face, different name)
          - If names are SAME: -5% (less suspicious, could be legitimate)

        Why this works:
        - You can't fake a face, so 90%+ match = definitely same person
        - If they also have different names, they're committing fraud
        - If they have same name too, might be data entry error (less priority)
        """
        print("="*70)
        print("SCENARIO 2: FAKE FACE DETECTION (90%+ Similarity)")
        print("="*70)
        print("Rule: Face similarity >= 90% = SAME PERSON = FAKE\n")

        fakes_scenario_2 = []

        # Get only records with faces
        faces_df = self.df[self.df['face_encoding'].notna()].reset_index(drop=True)
        encodings = list(faces_df['face_encoding'].values)

        if len(encodings) < 2:
            print(f"⚠ Not enough faces ({len(encodings)}) to compare\n")
            return fakes_scenario_2

        print(f"Comparing {len(encodings)} faces for 90%+ matches...\n")

        for i in range(len(encodings)):
            for j in range(i + 1, len(encodings)):
                # Calculate face similarity
                distance = face_recognition.face_distance([encodings[i]], encodings[j])[0]
                similarity_percent = (1 - distance) * 100

                # Threshold: 90% or higher
                if similarity_percent < 90.0:
                    continue

                row1 = faces_df.iloc[i]
                row2 = faces_df.iloc[j]

                # Calculate confidence score based on details
                confidence_score = similarity_percent
                details_evidence = []

                # Check name match for additional evidence
                if pd.notna(row1['name']) and pd.notna(row2['name']):
                    if row1['name'] != row2['name']:
                        # SUPER SUSPICIOUS: Same face but DIFFERENT names!
                        confidence_score += 5
                        details_evidence.append(f"DIFFERENT NAMES: '{row1['name']}' vs '{row2['name']}'")
                        likelihood = "🚨 CRITICAL FRAUD - SAME FACE, DIFFERENT NAMES"
                    else:
                        # Less suspicious: same face, same name too
                        confidence_score -= 5
                        details_evidence.append(f"SAME NAME: {row1['name']}")
                        likelihood = "LIKELY FRAUD - SAME PERSON"
                else:
                    likelihood = "LIKELY FRAUD - SAME PERSON"

                fakes_scenario_2.append({
                    'fraud_type': 'FAKE_FACE',
                    'card_1': row1['card_id'],
                    'card_2': row2['card_id'],
                    'name_1': row1['name'] if pd.notna(row1['name']) else 'N/A',
                    'name_2': row2['name'] if pd.notna(row2['name']) else 'N/A',
                    'father_1': row1['father_husband_name'] if pd.notna(row1['father_husband_name']) else 'N/A',
                    'father_2': row2['father_husband_name'] if pd.notna(row2['father_husband_name']) else 'N/A',
                    'face_similarity_percent': round(similarity_percent, 2),
                    'confidence_score': round(confidence_score, 1),
                    'details_evidence': ', '.join(details_evidence) if details_evidence else 'No matching details',
                    'likelihood': likelihood
                })

        print(f"✓ Found {len(fakes_scenario_2)} fake voters with identical faces\n")

        return fakes_scenario_2

    def detect_address_anomalies(self, suspicious_threshold=30):
        """
        SCENARIO 3: ADDRESS ANOMALIES

        Flag if too many voters share the same house number.

        Thresholds:
        - 10-20 voters: Normal (joint family)
        - 20-30 voters: Slightly suspicious (flag for review)
        - 30+ voters: HIGHLY SUSPICIOUS (likely fraud operation)
        """
        print("\n" + "="*70)
        print("SCENARIO 3: ADDRESS ANOMALY DETECTION")
        print("="*70)
        print(f"Rule: {suspicious_threshold}+ voters at same address = SUSPICIOUS\n")

        anomalies = []

        # Filter out records with missing house numbers
        df_with_house = self.df[self.df['house_number'].notna()].copy()

        if len(df_with_house) == 0:
            print("⚠ No house numbers available for analysis\n")
            return anomalies

        # Count voters per house
        house_counts = df_with_house.groupby('house_number').size().reset_index(name='voter_count')

        # Find suspicious addresses
        suspicious_houses = house_counts[house_counts['voter_count'] >= suspicious_threshold]

        print(f"Total unique addresses: {len(house_counts)}")
        print(f"Suspicious addresses (30+ voters): {len(suspicious_houses)}\n")

        for _, house in suspicious_houses.iterrows():
            house_num = house['house_number']
            count = house['voter_count']

            # Get all voters at this address
            voters_at_address = df_with_house[df_with_house['house_number'] == house_num]

            # Check for additional red flags
            unique_names = voters_at_address['name'].nunique()
            unique_fathers = voters_at_address['father_husband_name'].nunique()

            # Calculate suspicion level
            if count >= 50:
                risk_level = "CRITICAL"
            elif count >= 40:
                risk_level = "HIGH"
            else:
                risk_level = "MEDIUM"

            anomalies.append({
                'fraud_type': 'ADDRESS_ANOMALY',
                'house_number': house_num,
                'voter_count': count,
                'unique_names': unique_names,
                'unique_fathers': unique_fathers,
                'risk_level': risk_level,
                'cards': ', '.join(voters_at_address['card_id'].tolist()[:10]) + '...',  # First 10 cards
                'likelihood': f'{count} voters at one address - Likely fake voter operation'
            })

            print(f"  ⚠ House {house_num}: {count} voters")
            print(f"      Unique names: {unique_names}, Unique fathers: {unique_fathers}")
            print(f"      Risk: {risk_level}\n")

        print(f"✓ Found {len(anomalies)} suspicious addresses\n")
        return anomalies

    def detect_all_frauds(self):
        """Run all fraud detection scenarios including address anomalies"""
        print("\n" + "="*70)
        print("COMPREHENSIVE FRAUD DETECTION")
        print("="*70 + "\n")

        # Scenario 1: Duplicate details
        scenario_1 = self.detect_scenario_1_fake_details()

        # Scenario 2: Duplicate faces
        scenario_2 = self.detect_scenario_2_fake_face()

        # Scenario 3: Address anomalies (NEW)
        scenario_3 = self.detect_address_anomalies(suspicious_threshold=30)

        # Combine all frauds
        all_frauds = scenario_1 + scenario_2 + scenario_3

        # Print summary
        print("="*70)
        print("FRAUD DETECTION SUMMARY")
        print("="*70)
        print(f"\nScenario 1 (Duplicate Details):  {len(scenario_1)}")
        print(f"Scenario 2 (Duplicate Face):     {len(scenario_2)}")
        print(f"Scenario 3 (Address Anomalies):  {len(scenario_3)}")
        print(f"\n{'─'*70}")
        print(f"TOTAL FRAUDS DETECTED:           {len(all_frauds)}")
        print("="*70 + "\n")

        if all_frauds:
            return pd.DataFrame(all_frauds)
        else:
            return pd.DataFrame()

    def generate_report(self, frauds_df, output_dir="output/reports"):
        """Generate detailed fraud detection report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if frauds_df.empty:
            print("✅ NO FRAUDS DETECTED")
            print("   All voters appear to be legitimate!\n")
            return

        # Save CSV report
        report_file = output_dir / "fraud_detection_report.csv"
        frauds_df.to_csv(report_file, index=False, encoding='utf-8-sig')
        print(f"✓ Full report saved: {report_file}\n")

        # Display frauds
        print("="*70)
        print("DETECTED FRAUDS - DETAILED VIEW")
        print("="*70 + "\n")

        # Scenario 1: Fake details
        scenario_1 = frauds_df[frauds_df['fraud_type'] == 'FAKE_DETAILS']
        if not scenario_1.empty:
            print("🚨 SCENARIO 1: FAKE DETAILS (Blatant Lie)")
            print("   Same details, different voter IDs\n")
            print("─" * 70)
            for idx, row in scenario_1.iterrows():
                print(f"\n  FRAUD #{idx+1}:")
                print(f"    Card 1: {row['card_1']}")
                print(f"    Card 2: {row['card_2']}")
                print(f"    Name: {row['name']}")
                print(f"    Father/Husband: {row['father_husband']}")
                print(f"    Matching Fields: {row['matching_fields']}")
                print(f"    Confidence: {row['confidence']}")

        # Scenario 2: Fake face
        scenario_2 = frauds_df[frauds_df['fraud_type'] == 'FAKE_FACE']
        if not scenario_2.empty:
            print("\n\n🚨 SCENARIO 2: FAKE FACE (Clever Fraud)")
            print("   Same person, different voter details\n")
            print("─" * 70)
            for idx, row in scenario_2.iterrows():
                print(f"\n  FRAUD #{idx+1}:")
                print(f"    Card 1: {row['card_1']} - {row['name_1']}")
                print(f"    Card 2: {row['card_2']} - {row['name_2']}")
                print(f"    Face Similarity: {row['face_similarity_percent']}%")
                print(f"    Confidence Score: {row['confidence_score']}%")
                print(f"    Evidence: {row['details_evidence']}")
                print(f"    Assessment: {row['likelihood']}")

        # Scenario 3: Address anomalies
        scenario_3 = frauds_df[frauds_df['fraud_type'] == 'ADDRESS_ANOMALY']
        if not scenario_3.empty:
            print("\n\n🚨 SCENARIO 3: ADDRESS ANOMALIES (Suspicious Addresses)")
            print("   Too many voters at same address\n")
            print("─" * 70)
            for idx, row in scenario_3.iterrows():
                print(f"\n  ANOMALY #{idx+1}:")
                print(f"    House Number: {row['house_number']}")
                print(f"    Total Voters: {row['voter_count']}")
                print(f"    Unique Names: {row['unique_names']}")
                print(f"    Unique Fathers: {row['unique_fathers']}")
                print(f"    Risk Level: {row['risk_level']}")
                print(f"    Sample Cards: {row['cards']}")
                print(f"    Assessment: {row['likelihood']}")

        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Initialize detector
    detector = DuplicateDetectorFinal()

    # Detect all frauds
    frauds = detector.detect_all_frauds()

    # Generate and display report
    detector.generate_report(frauds)

    print("✓ Fraud detection complete!\n")
