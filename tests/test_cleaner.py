import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaner import clean_dataset


class CleanDatasetTests(unittest.TestCase):
    def test_normalizes_deduplicates_and_skips_invalid_ratings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "input.csv"
            output_path = directory / "processed" / "ratings.csv"
            report_path = directory / "processed" / "report.json"
            rows = [
                ["1", "10", "5", "  Film   Name ", "Drama | Comedy|Drama", "m", "02138"],
                ["1", "10", "5", "Film Name", "Drama|Comedy", "M", "02138"],
                ["2", "11", "6", "Invalid", "Drama", "F", "10001"],
            ]
            with input_path.open("w", encoding="utf-8", newline="") as source:
                writer = csv.writer(source)
                writer.writerow(
                    ["user_id", "movie_id", "rating", "title", "genres", "gender", "zip_code"]
                )
                writer.writerows(rows)

            report = clean_dataset(input_path, output_path, report_path)

            with output_path.open(encoding="utf-8", newline="") as cleaned:
                result = list(csv.DictReader(cleaned))
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["title"], "Film Name")
            self.assertEqual(result[0]["genres"], "Drama|Comedy")
            self.assertEqual(result[0]["gender"], "M")
            self.assertEqual(result[0]["zip_code"], "02138")
            self.assertEqual(report["rows_read"], 3)
            self.assertEqual(report["rows_written"], 1)
            self.assertEqual(report["invalid_rows_removed"], 1)
            self.assertEqual(report["exact_duplicates_removed"], 1)
            self.assertEqual(saved_report, report)

    def test_requires_recommender_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "input.csv"
            input_path.write_text("movie_id,rating\n1,4\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required columns: user_id"):
                clean_dataset(
                    input_path,
                    directory / "output.csv",
                    directory / "report.json",
                )

    def test_removes_invalid_ids_and_clears_invalid_optional_integers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "input.csv"
            output_path = directory / "output.csv"
            report_path = directory / "report.json"
            input_path.write_text(
                "user_id,movie_id,rating,timestamp\n1,10,4,not-a-timestamp\n0,11,5,123\n",
                encoding="utf-8",
            )

            report = clean_dataset(input_path, output_path, report_path)

            with output_path.open(encoding="utf-8", newline="") as cleaned:
                result = list(csv.DictReader(cleaned))

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["timestamp"], "")
            self.assertEqual(report["invalid_rows_removed"], 1)
            self.assertEqual(report["invalid_row_reasons"]["invalid_user_id"], 1)
            self.assertEqual(report["invalid_optional_values_cleared"]["timestamp"], 1)

    def _clean_text(self, text: str) -> tuple[list[dict], dict]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "input.csv"
            input_path.write_text(text, encoding="utf-8")
            output_path = directory / "output.csv"
            report = clean_dataset(input_path, output_path, directory / "report.json")
            with output_path.open(encoding="utf-8", newline="") as cleaned:
                return list(csv.DictReader(cleaned)), report

    def test_keeps_most_recent_rating_per_user_movie_pair(self) -> None:
        result, report = self._clean_text(
            "user_id,movie_id,rating,timestamp\n1,10,2,200\n1,10,5,300\n1,10,3,100\n2,10,4,150\n"
        )
        self.assertEqual(len(result), 2)
        kept = next(row for row in result if row["user_id"] == "1")
        self.assertEqual((kept["rating"], kept["timestamp"]), ("5", "300"))
        self.assertEqual(report["older_duplicate_ratings_removed"], 2)

    def test_harmonizes_conflicting_metadata_with_latest_row(self) -> None:
        result, report = self._clean_text(
            "user_id,movie_id,rating,timestamp,gender,title\n"
            "1,10,4,100,F,Old Title\n"
            "1,11,4,200,M,Other\n"
            "2,10,3,300,M,New Title\n"
        )
        self.assertEqual({row["gender"] for row in result if row["user_id"] == "1"}, {"M"})
        self.assertEqual({row["title"] for row in result if row["movie_id"] == "10"}, {"New Title"})
        self.assertEqual(report["inconsistent_metadata_harmonized"], {"users": 1, "movies": 1})

    def test_clears_implausible_timestamps(self) -> None:
        result, report = self._clean_text(
            "user_id,movie_id,rating,timestamp\n1,10,4,-5\n1,11,4,99999999999\n1,12,4,978300760\n"
        )
        self.assertEqual([row["timestamp"] for row in result], ["", "", "978300760"])
        self.assertEqual(report["invalid_optional_values_cleared"]["timestamp"], 2)
        self.assertEqual(report["invalid_row_reasons"]["invalid_rating"], 0)


if __name__ == "__main__":
    unittest.main()
