from utils import frame_to_hms


class Logger():
    def __init__(self):
        self.stats = {
                'total': {'grab': 0,
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
        self.stats['total'][event] += 1  
        self.stats['total']['events'].append(f"{event[0]}{self.stats['total'][event]}") 
        self.buffer['events'].append(f"{event[0]}{self.stats['total'][event]}") 
        self.logs.append(
            {'message':f'{event} happened at: {logging_time}',
                'frames_left':250,
                'total_frames':250
                })
        if event == 'transition':
            self.stats['per_bag'].append(self.buffer)
            self.buffer = self.reset_buffer()

    def update_logs(self):
        self.logs = list(filter(lambda log: log['frames_left'] > 0, self.logs))