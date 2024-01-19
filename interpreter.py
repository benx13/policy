

class Interpreter():
    def __init__(self, sequence) -> None:
        self.sequence = sequence
        self.original_sequence = [i for i in self.sequence]
        self.unduped = []
        self.clean_sequence = []
        self.sequence_remains = []
        self.handle_dict = {'gbm':0,
                            'gmf':0,
                            'missing':0,
                            'extra':0,
                            'finished_probability': 0
                            }
    
    def postprocess_sequence(self):
        if (len(self.sequence)>3):
            self.pop_dupes()
            self.sequence = [i for i in self.unduped]
            self.sequence_remains = [i for i in self.sequence]
            self.build_sequence()

            self.sequence_remains = [i[0] for i in self.sequence_remains]
            #print(self.handle_dict)
            #print(f'remains before: {self.sequence_remains}')
            #print(f'clean before: {self.clean_sequence}')
            #print('-----------------')
            self.process_remains()
            self.handle_dict['extra'] += len(self.sequence_remains)
            #print(f'remains after: {self.sequence_remains}')
            #print(f'remains after: {self.clean_sequence}')
            #print(self.handle_dict)
            self.handle_dict['finished_probability'] = (self.handle_dict['gmf'] + self.handle_dict['gbm'])/4 - (self.handle_dict['extra'] + self.handle_dict['missing']) * 0.05

        else:
            self.sequence_remains = self.sequence
            self.handle_dict['extra'] = len(self.sequence_remains)

        return self.clean_sequence, self.sequence_remains, self.handle_dict, self.original_sequence
    
    def pop_dupes(self):
        for i in range(len(self.sequence) - 1):
            event1, _ = self.parse_element(self.sequence[i])
            event2, _ = self.parse_element(self.sequence[i + 1])
            if event2 != event1:
                self.unduped.append(self.sequence[i])
            else:
                self.handle_dict['extra'] += 1
                self.reset_remaining_sequence_index_decrease(event1, i+1)
                continue
        event1, _ = self.parse_element(self.sequence[-2])
        event2, _ = self.parse_element(self.sequence[-1])
        if event2 != event1:
            self.unduped.append(self.sequence[-1])
        return self.unduped
    def parse_element(self, event):
        event_type = event[0]
        event_value = int(event[1:])
        return event_type, event_value

    def reset_remaining_sequence_index_decrease(self, event, start_index):
        for i in range(start_index, len(self.sequence)):
            event_, x = self.parse_element(self.sequence[i])
            if event_ == event and event_ in ['g', 'm']:
                self.sequence[i] = f'{event}{x-1}'
            if event_ == event and event_ in ['f', 'b']:
                self.sequence[i] = f'{event}{x-2}'

    def reset_remaining_sequence_index_increase(self, event, start_index):
        for i in range(start_index, len(self.sequence)):
            event_, x = self.parse_element(self.sequence[i])
            if event_ == event and event_ in ['g', 'm']:
                self.sequence[i] = f'{event}{x-1}'
            if event_ == event and event_ in ['f', 'b']:
                self.sequence[i] = f'{event}{x-2}'


    def build_sequence(self):
        for i in range(len(self.sequence) - 2):
            event1, x = self.parse_element(self.sequence[i])
            event2, y = self.parse_element(self.sequence[i + 1])
            event3, z = self.parse_element(self.sequence[i + 2])
            #print(f'{event1}{x} -> {event2}{y} -> {event3}{z}')   
            if [event1, event2, event3] == ['g', 'b', 'm']:# and x == y and y == z:
                self.handle_dict['gbm'] += 1
                self.clean_sequence.append(((event1, x), (event2, y), (event3, z)))
                self.sequence_remains.remove(f'{event1}{x}')
                self.sequence_remains.remove(f'{event2}{y}')
                self.sequence_remains.remove(f'{event3}{z}')
            if [event1, event2, event3] == ['g', 'm', 'f']:# and x == y and y == z:
                self.handle_dict['gmf'] += 1
                self.clean_sequence.append(((event1, x), (event2, y), (event3, z)))
                self.sequence_remains.remove(f'{event1}{x}')
                self.sequence_remains.remove(f'{event2}{y}')
                self.sequence_remains.remove(f'{event3}{z}')

    def process_remains(self):
        if(len(self.sequence_remains) <2):
            return
        missing_events = (2 - self.handle_dict['gbm']) * [['g', 'b', 'm']] + (2 - self.handle_dict['gmf']) * [['g', 'm', 'f']]
        #print(missing_events)

        for events in missing_events:
            sequence_remains_set = set(self.sequence_remains)
            #print(f'sequence_remains_set--->{sequence_remains_set}')
            #print(f'common elems -->{set(events) - sequence_remains_set}')
            if len(sequence_remains_set) >2:
                if (set(events).issubset(sequence_remains_set)):
                    #print('@@@@')
                    self.clean_sequence.append(((events[0], -1), (events[1], -1), (events[2], -1)))
                    self.sequence_remains.remove(events[0])
                    self.sequence_remains.remove(events[1])
                    self.sequence_remains.remove(events[2])
                    self.handle_dict[''.join(events)]+=1
                if(len(set(events) - sequence_remains_set) == 1):
                    #print('%%%%')
                    self.clean_sequence.append(((events[0], -1), (events[1], -1), (events[2], -1)))
                    self.sequence_remains.append(list(set(events) - sequence_remains_set)[0])
                    self.sequence_remains.remove(events[0])
                    self.sequence_remains.remove(events[1])
                    self.sequence_remains.remove(events[2])
                    self.handle_dict[''.join(events)] += 1
                    self.handle_dict['missing'] += 1
            if len(sequence_remains_set)  == 2:
                if(len(set(events) - sequence_remains_set) == 1):
                    #print('%%%%')
                    self.clean_sequence.append(((events[0], -1), (events[1], -1), (events[2], -1)))
                    self.sequence_remains.append(list(set(events) - sequence_remains_set)[0])
                    self.sequence_remains.remove(events[0])
                    self.sequence_remains.remove(events[1])
                    self.sequence_remains.remove(events[2])
                    self.handle_dict[''.join(events)] += 1
                    self.handle_dict['missing'] += 1

'''

events = [event.split(' -> ') for event in 
['g1 -> m1 -> f1 -> g2 -> b2 -> m2 -> g3 -> m3 -> f3 -> g4 -> b4 -> m4', 
 'b2 -> m1 -> f1 -> b4 -> m2',
 'g1 -> m1 -> f1 -> g2 -> b2 -> m2 -> g3 -> m3 -> g4 -> f3 -> g5 -> b4 -> m4', 
 'g1 -> f1 -> g2 -> b2 -> m1 -> g3 -> m2 -> f3 -> g4 -> g5 -> b4 -> m3',
 'g1 -> m1 -> f1 -> g2 -> b2 -> m2 -> m3 -> g3 -> m4 -> f3 -> g4 -> b4 -> m5',
 'm1 -> f1 -> g2 -> b2 -> m2 -> g3 -> m3 -> f3 -> m4 -> b4 -> m5',
 'g1 -> m1 -> f1 -> g2 -> b2 -> m2 -> g3 -> g4 -> m3 -> f3 -> g5 -> g6 -> b4 -> m4',
 'g1 -> m1 -> f1 -> g2 -> b2 -> m2 -> f3 -> g3 -> m3 -> f5 -> g4 -> b4 -> m4',
 'g1 -> m1 -> f1 -> m2 -> b2 -> m3 -> g3 -> m4 -> f3 -> g4 -> b4 -> m5',
 'g1 -> m1 -> f1 -> g2 -> b2 -> m2 -> g3 -> m3 -> f3 -> g4 -> b4 -> m4 -> g5']
]

for i in events[:]:
    print('--------------------------------------------------------')
    clean, remains, count, og = Interpreter(i).postprocess_sequence()

    print(f'og:         --->  {og}')
    print(f'clean:      --->  {clean}')
    print(f'remains:    --->  {remains}')
    print(f'ditc:       --->  {count}')

'''