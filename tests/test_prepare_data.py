import unittest

from prepare_data import parse_record


class ParseRecordTests(unittest.TestCase):
    def test_regular_record(self):
        row = parse_record('18.0 8 307.0 130.0 3504. 12.0 70 1 "chevrolet chevelle malibu"')
        self.assertEqual(row, [
            "chevrolet chevelle malibu", "18.0", "8", "307.0", "130.0",
            "3504.", "12.0", "70", "usa",
        ])

    def test_missing_horsepower(self):
        row = parse_record('25.0 4 98.00 ? 2046. 19.0 71 1 "ford pinto"')
        self.assertEqual(row[4], "")

    def test_unexpected_origin_rejected(self):
        with self.assertRaises(ValueError):
            parse_record('25.0 4 98.00 80 2046. 19.0 71 9 "ford pinto"')


if __name__ == "__main__":
    unittest.main()
