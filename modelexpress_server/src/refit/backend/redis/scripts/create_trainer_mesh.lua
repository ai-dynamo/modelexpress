-- Reserve the idempotency key and create the mesh atomically.
-- KEYS: mesh hash, idempotency key
-- ARGV: mesh_id, model_name, logical_shards JSON, workers JSON, trainer_role
local existing = redis.call('GET', KEYS[2])
if existing then
  return 'EXISTING:' .. existing
end
if redis.call('EXISTS', KEYS[1]) == 1 then
  return 'COLLISION'
end
local workers = cjson.decode(ARGV[4])
for index = 3, #KEYS do
  if redis.call('EXISTS', KEYS[index]) == 0 then
    return 'WORKER_NOT_FOUND'
  end
  if redis.call('HGET', KEYS[index], 'model_name') ~= ARGV[2]
      or redis.call('HGET', KEYS[index], 'role') ~= ARGV[5] then
    return 'WORKER_MISMATCH'
  end
  local worker_id = redis.call('HGET', KEYS[index], 'worker_id')
  if not workers[worker_id]
      or redis.call('HGET', KEYS[index], 'refit_endpoint') ~= workers[worker_id].metadata_endpoint then
    return 'WORKER_MISMATCH'
  end
end
redis.call('HSET', KEYS[1],
  'mesh_id', ARGV[1],
  'model_name', ARGV[2],
  'generation', 1,
  'logical_shards', ARGV[3],
  'workers', ARGV[4],
  'initial_workers', ARGV[4])
redis.call('SET', KEYS[2], ARGV[1])
return 'CREATED'
