'''
This script is called when the module is disabled.
It stops the BrainFlow stream and releases the session to free up resources.
'''
print('##################################################')
print('CLOSING BRAINFLOW EEG MODULE')

try:
    if board.is_prepared():
        print('Stopping BrainFlow stream...')
        board.stop_stream()
        print('Releasing BrainFlow session...')
        board.release_session()
        print('BrainFlow session closed.')
    else:
        print('BrainFlow session was not prepared. No action needed.')
except NameError:
    print("Error: 'board' object not found. Was the constructor run correctly?")
except Exception as e:
    print(f"An error occurred while closing the BrainFlow session: {e}")
