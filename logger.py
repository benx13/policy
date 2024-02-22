from utils import *
import json
from interpreter import Interpreter

class Logger():
    def __init__(self):
        self.stats = {
                'BAG_COUNT': 0,
                'bag_count_based_on_total_number_of_events': 0,
                'bag_count_based_on_order_of_events': 0,
                'normalized_bag_count': 0,
                'average_time_per_bag': 0,
                't2t_mean':0,
                'total_machine_time': 0,
                'average_machine_time_per_bag': 0,
                'total_bags_according_to_machine_time': 0,
                'bag_times':[],
                'bag_machine_times':[],
                't2t_list':[],
                'transitions':[],
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
        'backward': 0,
        'machine': 0,
        'transition': 0,
        'events': [],
        'bag_time': 0,
        'bag_machine_time': 0
        }
        self.logs = []
        self.transition_id = 0

    def reset_buffer(self):
        self.buffer = {'grab': 0,
            'forward': 0,
            'backward':0,
            'machine':0,
            'transition':0,
            'events': [],
            'bag_time':0,
            'bag_machine_time': 0,
            't2t': 0
            }

    def update(self, event, logging_time):
        if event != 'transition':
            self.buffer[event] += 1
            self.buffer['events'].append({'event':f"{event[0]}{self.buffer[event]}", 'time':logging_time}) 
        self.stats['total'][event] += 1  
        self.stats['total']['events'].append({'event':f"{event[0]}{self.stats['total'][event]}", 'time':logging_time}) 
        self.logs.append(
            {'message':f'{event} happened at: {logging_time}',
                'frames_left':250,
                'total_frames':250
                })
        if event == 'transition':
            self.stats['transitions'].append(logging_time)
            self.buffer[event] += 1
            self.buffer['events'].append({'event':f"{event[0]}{self.buffer[event]}", 'time':logging_time})
            self.buffer['id'] = self.transition_id

            if self.transition_id > 0:
                #t2t = hms_difference(self.stats['transitions'][self.transition_id], self.stats['transitions'][self.transition_id-1])
                t2t = '00:00:00'
                self.buffer['t2t'] = t2t
                self.stats['t2t_list'].append(t2t)


            clean_sequence, sequence_remains, decision, _, next_iter = Interpreter([i['event'] for i in self.buffer['events']]).postprocess_sequence()
            
            if(next_iter):
                self.process_next_iter(decision, clean_sequence, sequence_remains)
            else:
                self.update_stats(decision, clean_sequence, sequence_remains)
                self.reset_buffer()
                self.transition_id += 1

    def update_logs(self):
        self.logs = list(filter(lambda log: log['frames_left'] > 0, self.logs))

    def update_flow(self, flow):
        self.stats['total_machine_time'] += 1
        self.buffer['bag_machine_time'] += 1

    def save_results(self, name):
        with open(f'{name}.json', 'w') as fp:
            json.dump(self.stats, fp)
    def update_stats(self, decision, clean_sequence, sequence_remains):
        self.stats['BAG_COUNT'] += 1 if decision['finished_probability'] > 0.4 else 0
        self.stats['bag_count_based_on_order_of_events'] += decision['finished_probability']
        self.stats['bag_count_based_on_total_number_of_events'] = (self.stats['total']['grab']/4 + self.stats['total']['machine']/4 + self.stats['total']['forward']/2 + self.stats['total']['backward']/2)/4
        self.stats['normalized_bag_count'] = (self.stats['bag_count_based_on_order_of_events'] + self.stats['bag_count_based_on_order_of_events'] + self.stats['BAG_COUNT'])/3
        self.buffer['postprocessed_events'] = {
            'clean_sequence': clean_sequence,
            'sequence_remains': sequence_remains,
            'decision': decision
        }
        self.buffer['bag_time'] = hms_difference(self.buffer['events'][0]['time'], self.buffer['events'][-1]['time'])
        self.stats['bag_times'].append(self.buffer['bag_time'])
        self.stats['average_time_per_bag'] = hms_mean([i for i in self.stats['bag_times'] if int(i[4]) > 2])
        self.stats['t2t_mean'] = hms_mean([i  for i in self.stats['t2t_list'] if int(i[4]) > 2])

        self.stats['per_bag'].append(self.buffer)
        self.stats['average_machine_time_per_bag'] = np.mean([bag['bag_machine_time']  for bag in self.stats['per_bag'] if bag['postprocessed_events']['decision']['finished_probability'] > 0.5])
        #self.stats['total_bags_according_to_machine_time'] = self.stats['total_machine_time'] / self.stats['average_machine_time_per_bag']
    def process_next_iter(self, decision, clean_sequence, sequence_remains):
            last_event = f'{clean_sequence[-1][-1][0]}{clean_sequence[-1][-1][1]}'
            last_event_index = [i['event'] for i in self.buffer['events']].index(last_event)
            last_event_time = self.buffer['events'][last_event_index]['time']

            tmp = self.buffer['events']
            left = self.buffer['events'][:last_event_index]
            left.append({'event':f"t1", 'time':last_event_time})
            right = self.buffer['events'][last_event_index:]
            self.buffer['events'] = left
            self.update_stats(decision, clean_sequence, sequence_remains)
            self.reset_buffer()
            self.transition_id += 1

            clean_sequence, sequence_remains, decision, _, next_iter = Interpreter([i['event'] for i in right]).postprocess_sequence()
            self.buffer['events'] = right
            self.update_stats(decision, clean_sequence, sequence_remains)
            self.reset_buffer()
            self.transition_id += 1