import click
import pandas as pd

@click.command()
@click.option('-sf', '--source_filepath', help='source file', required=True)
@click.option('-df', '--destination_filepath', help='destination file', required=True)
def clean_twitter_data(source_filepath, destination_filepath):
    values = []
    with open(source_filepath) as f:
        
        for i, line in enumerate(f):
            if i == 0:
                colnames = line
            else:
                values.append(line.split(',')[:4])
    data = pd.DataFrame(values, columns = colnames.split(','))
    data.to_csv(destination_filepath, index=None)   

if __name__ == '__main__':
    clean_twitter_data()