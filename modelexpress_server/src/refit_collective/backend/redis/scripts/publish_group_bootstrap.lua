-- Atomically record one lane leader's ncclUniqueId, then re-evaluate readiness.
--
-- KEYS[1]: group hash
-- KEYS[2]: participants hash
-- KEYS[3]: reported plan digests hash
-- KEYS[4]: the lane hash being published
-- KEYS[5..]: every lane hash, for the readiness sweep
-- ARGV[1]: epoch the caller generated the identifier for
-- ARGV[2]: worker_id
-- ARGV[3]: nccl_unique_id, hex encoded
-- ARGV[4]: slot_id assigned rank 0 in this lane
--
-- The helpers come from common.lua, which redis.rs prepends.

local epoch = tonumber(redis.call('HGET', KEYS[1], 'epoch'))
if not epoch then
  return 'NOTFOUND'
end

if tonumber(ARGV[1]) ~= epoch then
  return 'STALE:' .. epoch
end

local model_name = redis.call('HGET', KEYS[1], 'model_name')
local leader_record = redis.call('HGET', KEYS[2], ARGV[4])
local leader_worker, leader_role, leader_index, joined_epoch =
  parse_participant(leader_record)
if not leader_worker or leader_worker ~= ARGV[2] or tonumber(joined_epoch) ~= epoch
    or not registration_matches(leader_worker, leader_role, model_name) then
  return 'NOTLEADER'
end

local existing_epoch = tonumber(redis.call('HGET', KEYS[4], 'bootstrap_epoch'))
if existing_epoch == epoch then
  local existing_publisher = redis.call('HGET', KEYS[4], 'published_by')
  local existing_id = redis.call('HGET', KEYS[4], 'nccl_unique_id')
  if existing_publisher ~= ARGV[2] or existing_id ~= ARGV[3] then
    return 'CONFLICT'
  end
else
  redis.call('HSET', KEYS[4],
    'nccl_unique_id', ARGV[3],
    'bootstrap_epoch', epoch,
    'published_by', ARGV[2])
end

local expected = tonumber(redis.call('HGET', KEYS[1], 'expected_total'))
local ready = evaluate_readiness(
  KEYS[1], KEYS[2], KEYS[3], key_range(5, #KEYS), expected, epoch, model_name)

local state = ready and 'READY' or 'FORMING'
redis.call('HSET', KEYS[1], 'state', state)

return 'OK:' .. epoch .. ':' .. state
