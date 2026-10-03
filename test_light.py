import time
print('1. Import...', flush=True)
from jobs.cnn_3d.attention_job import AttentionCNN3DJob
print('2. Job...', flush=True)
job = AttentionCNN3DJob()
print('3. Params...', flush=True)
theta = job.init_params()
opt = job.init_opt_state()
print('Params:', job.n_params(), flush=True)

print('4. Test 1 sample...', flush=True)
task = job.make_task(1, 0, seed=42, epsilon=0.1)
task['batch_size'] = 4
t0 = time.time()
g, n, m = job.compute_gradient(theta, task)
print('OK %.1fs  loss=%.3f  acc=%.3f' % (time.time()-t0, m['loss'], m['accuracy']), flush=True)

print('5. evaluate...', flush=True)
print(job.evaluate(theta, 30), flush=True)
print('DONE', flush=True)
