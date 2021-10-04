import numpy as np
import watchdog.events
from watchdog.observers import Observer
from data_anal_tools import extract_data_from_csv, slope
from analysis_method import evaluate_performance

import time
  
# gloabl variables that will be updated once a folder is created
isWatch = True
res = []

# file watch params
final_peak = 0

# check duplication
duplicate = []

class HPLC_watch:

	def __init__(self, watch_dir, file_type): 
		self.observer = Observer()
		self.watch_dir = watch_dir
		self.file_type = file_type

	def run(self): 
		global isWatch, final_peak, duplicate, res

		print('monitoring............')
		event_handler = Handler(self.file_type)
		self.observer.schedule(event_handler, path=self.watch_dir, recursive=True)
		self.observer.start()

		while isWatch:
			time.sleep(1)
		self.observer.stop()
		self.observer.join()

		# reset global params
		isWatch = True
		all_peak_areas, duplicate = [], []
		index = 0

		return res

class Handler(watchdog.events.PatternMatchingEventHandler):
	
	def __init__(self, file_type):
		watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=file_type, ignore_directories=True, case_sensitive=False)

	def on_created(self, event): 
		global isWatch, final_peak, duplicate, res

		filepath = event.src_path

		# to see if watchdog is monitoring duplicating events
		duplicate.append((filepath))
		if len(duplicate) > 1:
			if duplicate[-1] == duplicate[-2]: return

		# back trace to the parent dir
		temp_path = list(filepath)[::-1]
		ind = np.where(np.array(temp_path) == '/')[0][1]
		parent_folder = filepath[:-ind]

		# handle the lag between event init and event completion
		try:
			res = evaluate_performance(parent_folder, keyword='rxn', range_of_interest=[1, 2])
		except PermissionError:
			time.sleep(1)
			res = evaluate_performance(parent_folder, keyword='rxn', range_of_interest=[1, 2])

		isWatch = False
		print('found target peak areas: ')
		print(res)


if __name__ == "__main__":
	watch = HPLC_watch('.', ['*.ch'])
	watch.run()