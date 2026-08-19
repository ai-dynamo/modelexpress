-- Atomically record one lane's ncclUniqueId, then re-evaluate readiness.
--
-- KEYS[1]: group hash
-- KEYS[2]: participants hash
-- KEYS[3]: reported plan digests hash
-- KEYS[4]: the lane hash being published
-- KEYS[5..]: every lane hash, for the readiness sweep
-- ARGV[1]: epoch the caller generated the identifier for
-- ARGV[2]: worker_id
-- ARGV[3]: nccl_unique_id, hex encoded
--
-- Returns:
--   OK:<epoch>:<state>
--   NOTFOUND           no such group
--   STALE:<epoch>      the caller's epoch is not the group's current epoch

local epoch = tonumber(redis.call('HGET', KEYS[1], 'epoch'))
if not epoch then
  return 'NOTFOUND'
end

-- Rejected rather than applied: a late publish carrying a superseded epoch
-- would otherwise overwrite the identifier the current membership is using,
-- and every rank would then initialize a communicator nobody else is in.
if tonumber(ARGV[1]) ~= epoch then
  return 'STALE:' .. epoch
end

redis.call('HSET', KEYS[4],
  'nccl_unique_id', ARGV[3],
  'bootstrap_epoch', epoch,
  'published_by', ARGV[2])

local lane_count = #KEYS - 4
local admitted = redis.call('HLEN', KEYS[2])
local expected = tonumber(redis.call('HGET', KEYS[1], 'expected_total'))
local ready = expected ~= nil and admitted == expected

if ready then
  for i = 1, lane_count do
    local stamped = redis.call('HGET', KEYS[4 + i], 'bootstrap_epoch')
    if not stamped or tonumber(stamped) ~= epoch then
      ready = false
      break
    end
  end
end

if ready then
  local digest = redis.call('HGET', KEYS[1], 'plan_digest')
  local reported = redis.call('HVALS', KEYS[3])
  for i = 1, #reported do
    if reported[i] ~= digest then
      ready = false
      break
    end
  end
end

local state = ready and 'READY' or 'FORMING'
redis.call('HSET', KEYS[1], 'state', state)

return 'OK:' .. epoch .. ':' .. state
