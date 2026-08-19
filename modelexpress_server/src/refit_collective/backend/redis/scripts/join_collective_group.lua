-- Atomically admit one worker to a collective group, creating it on first
-- contact, and re-evaluate readiness.
--
-- KEYS[1]: group hash
-- KEYS[2]: participants hash          (slot_id -> worker_id|role|index|partition)
-- KEYS[3]: reported plan digests hash (slot_id -> digest)
-- KEYS[4..3+lane_count]: one lane hash per lane, reshard lanes then broadcast
-- ARGV[1]:  group_id
-- ARGV[2]:  model_name
-- ARGV[3]:  source_partition_count
-- ARGV[4]:  expected participant total
-- ARGV[5]:  slot_id
-- ARGV[6]:  worker_id
-- ARGV[7]:  role
-- ARGV[8]:  index_in_role
-- ARGV[9]:  source_partition ('' for generators)
-- ARGV[10]: plan_digest
-- ARGV[11]: plan_source_worker_id ('' when this worker does not serve it)
-- ARGV[12]: plan_source_endpoint
-- ARGV[13]: plan_source_digest
-- ARGV[14]: expected_trainer_slots, newline separated
-- ARGV[15]: expected_generator_slots, newline separated
-- ARGV[16]: created_at_unix_ms
--
-- Returns: OK:<epoch>:<state>
--
-- Epoch bumps on any change that invalidates a cached communicator or a cached
-- plan: a newly admitted slot, a slot presenting a new worker generation, or a
-- different plan digest. Bumping clears every lane's bootstrap identifier in
-- the same transaction, because an identifier from the previous epoch names a
-- communicator whose world size no longer matches the membership.

local lane_count = #KEYS - 3
local epoch = tonumber(redis.call('HGET', KEYS[1], 'epoch'))
local changed = false

if not epoch then
  epoch = 1
  redis.call('HSET', KEYS[1],
    'group_id', ARGV[1],
    'model_name', ARGV[2],
    'source_partition_count', ARGV[3],
    'expected_total', ARGV[4],
    'expected_trainer_slots', ARGV[14],
    'expected_generator_slots', ARGV[15],
    'plan_digest', ARGV[10],
    'epoch', epoch,
    'state', 'FORMING',
    'plan_source_worker_id', '',
    'plan_source_endpoint', '',
    'plan_source_digest', '',
    'created_at_unix_ms', ARGV[16])
else
  local existing = redis.call('HGET', KEYS[2], ARGV[5])
  local incoming = ARGV[6] .. '|' .. ARGV[7] .. '|' .. ARGV[8] .. '|' .. ARGV[9]
  -- Only a *replacement* invalidates anything. Admitting an expected slot for
  -- the first time cannot invalidate a communicator or a plan, because none
  -- exists yet -- and bumping there would churn the epoch once per participant
  -- during formation, leaving every worker but the last holding a stale one.
  if existing and existing ~= incoming then
    changed = true
  end
  if redis.call('HGET', KEYS[1], 'plan_digest') ~= ARGV[10] then
    changed = true
  end
end

redis.call('HSET', KEYS[2], ARGV[5], ARGV[6] .. '|' .. ARGV[7] .. '|' .. ARGV[8] .. '|' .. ARGV[9])
redis.call('HSET', KEYS[3], ARGV[5], ARGV[10])

if ARGV[11] ~= '' then
  redis.call('HSET', KEYS[1],
    'plan_source_worker_id', ARGV[11],
    'plan_source_endpoint', ARGV[12],
    'plan_source_digest', ARGV[13])
end

if changed then
  epoch = epoch + 1
  redis.call('HSET', KEYS[1], 'epoch', epoch, 'plan_digest', ARGV[10], 'state', 'FORMING')
  for i = 1, lane_count do
    redis.call('DEL', KEYS[3 + i])
  end
end

-- Readiness: every expected slot admitted, every lane bootstrapped for THIS
-- epoch, and every participant reporting the same plan digest.
local admitted = redis.call('HLEN', KEYS[2])
local ready = admitted == tonumber(ARGV[4])

if ready then
  for i = 1, lane_count do
    local stamped = redis.call('HGET', KEYS[3 + i], 'bootstrap_epoch')
    if not stamped or tonumber(stamped) ~= epoch then
      ready = false
      break
    end
  end
end

if ready then
  local digests = redis.call('HVALS', KEYS[3])
  for i = 1, #digests do
    if digests[i] ~= ARGV[10] then
      ready = false
      break
    end
  end
end

local state = ready and 'READY' or 'FORMING'
redis.call('HSET', KEYS[1], 'state', state)

return 'OK:' .. epoch .. ':' .. state
