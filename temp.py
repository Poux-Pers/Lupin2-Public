import os
import sys
sys.stdout = open(os.getcwd()+'\\models\\logs\\temp.txt', 'w')

for time in ['Year', 'Month']:
    for region in ['Region', 'Month']:
        print('for ['+time+'] for ['+region+']')


for region in ['Region', 'Month']:
    for unit in ['Operating Unit', 'Segment', 'Service line']:        
        print('for ['+region+'] for ['+unit+']')


for time in ['Year', 'Month']:
    for unit in ['Operating Unit', 'Segment', 'Service line']:
        print('for ['+time+'] for ['+unit+']')

            
sys.stdout.close()