import csv
import os


class CsvManager:
    def __init__(self, file_name):
        self.file_name = file_name

    def get_data(self):
        with open(self.file_name + ".csv") as csvfile:
            data = []
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
            return data

    def create_append_data_to_csv(self, data, headers):
        with open(self.file_name+"_result.csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=",", lineterminator="\r", fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
