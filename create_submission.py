import tarfile
import os

BASE = os.path.dirname(os.path.abspath(__file__))
changed = os.getcwd() != BASE
if changed:
    prev = os.getcwd()
    os.chdir(BASE)

t = tarfile.open('submission.tar.gz', 'w:gz')
t.add('main.py')
t.add('deck.csv')
t.add('model.pt')
t.close()

if changed:
    os.chdir(prev)

print(f'Created submission.tar.gz ({os.path.getsize(BASE + "/submission.tar.gz") / 1024:.1f} KiB)')
