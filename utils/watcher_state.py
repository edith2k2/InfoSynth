class WatcherState:
    watcher_started = False
    watched_paths = []
    observer = None
    files_changed = False
    max_results = 5
    max_chain_of_thought_search_steps = 1


watcher_state = WatcherState()
