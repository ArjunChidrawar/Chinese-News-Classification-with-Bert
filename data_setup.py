import pandas as pd
import lzma
from io import StringIO

input_xz_file = 'test.csv.xz'
output_csv_file = 'input.csv'

with lzma.open(input_xz_file, 'rb') as xz_file:
    # Read the uncompressed data
    uncompressed_data = xz_file.read()

# Decode the bytes data to string
decoded_data = uncompressed_data.decode()

# Assuming the decoded data is in CSV format and split into lines
data_lines = decoded_data.splitlines()

# Create a StringIO object to make it compatible with pd.read_csv()
csv_data = StringIO('\n'.join(data_lines))

# Read the CSV data into a DataFrame
df = pd.read_csv(csv_data)

# Save the DataFrame to a CSV file
df.to_csv(output_csv_file, index=False)



def parse_custom_csv(input_file):
    rows = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Split each line at the first comma
            index = line.find(',')
            if index != -1:
                label = line[:index].strip()
                content = line[index + 1:].strip()
                rows.append((label, content))
    
    # Create a DataFrame from the parsed rows
    df = pd.DataFrame(rows, columns=['Label', 'Content'])
    return df

input_file_path = 'input.csv'
output_file_path = 'reviews.csv'

df = parse_custom_csv(input_file_path)
df.to_csv(output_file_path, index=False)

#Dataset is made up of 5 classes for 5 different types of news:
#1: Mainland China Politics
#2: HongKong/Macau Politics
#3: Taiwan Politics
#4: Military News
#5: Society News