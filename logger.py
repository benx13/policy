from utils import frame_to_hms
import json
from interpreter import Interpreter

class Logger():
    def __init__(self):
        self.stats = {
                'BAG_COUNT': 0,
                'bag_count_based_on_total_number_of_events': 0,
                'bag_count_based_on_order_of_events': 0,
                'normalized_bag_count': 0,
                'total': {  'grab': 0,
                            'forward': 0,
                            'backward':0,
                            'machine':0,
                            'transition':0,
                            'events':[]
                },
                'per_bag':[] 
        }
        self.buffer = {'grab': 0,
        'forward': 0,
        'backward':0,
        'machine':0,
        'events': []
        }
        self.logs = []
        self.transition_id = 0

    def reset_buffer(self):
        self.buffer = {'grab': 0,
            'forward': 0,
            'backward':0,
            'machine':0,
            'events': []
            }

    def update(self, event, logging_time):
        if event != 'transition':
            self.buffer[event] += 1
            self.buffer['events'].append(f"{event[0]}{self.buffer[event]}") 
        self.stats['total'][event] += 1  
        self.stats['total']['events'].append(f"{event[0]}{self.stats['total'][event]}") 
        self.logs.append(
            {'message':f'{event} happened at: {logging_time}',
                'frames_left':250,
                'total_frames':250
                })
        if event == 'transition':
            self.buffer['id'] = self.transition_id
            clean_sequence, sequence_remains, decision, _ = Interpreter(self.buffer['events']).postprocess_sequence()
            self.stats['BAG_COUNT'] += 1 if decision['finished_probability'] > 0.5 else 0
            self.stats['bag_count_based_on_order_of_events'] += decision['finished_probability']
            self.stats['bag_count_based_on_total_number_of_events'] = (self.stats['total']['grab']/4 + self.stats['total']['machine']/4 + self.stats['total']['forward']/2 + self.stats['total']['backward']/2)/4
            self.stats['normalized_bag_count'] = (self.stats['bag_count_based_on_order_of_events'] + self.stats['bag_count_based_on_order_of_events'] + self.stats['BAG_COUNT'])/3
            self.buffer['postprocessed_events'] = {
                'clean_sequence': clean_sequence,
                'sequence_remains': sequence_remains,
                'decision': decision
            }
            self.stats['per_bag'].append(self.buffer)
            self.reset_buffer()
            self.transition_id += 1

    def update_logs(self):
        self.logs = list(filter(lambda log: log['frames_left'] > 0, self.logs))

    def save_results(self):
        with open('output.json', 'w') as fp:
            json.dump(self.stats, fp)