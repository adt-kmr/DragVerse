using UnityEngine;
using UnityEngine.AI;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class BuggyAgent : Agent
{
    [Header("References")]
    public BuggyController controller;
    public MultiGoalPathHelper pathHelper;

    private Rigidbody rb;
    private int currentWaypointIndex;
    private Vector3 startPosition;
    private Quaternion startRotation;

    [Header("Tuning")]
    public float waypointReachDistance = 1.5f; // raised - matches buggy's real turning radius
    public float liftReachBonus = 3f;
    public float allLiftsBonus = 15f;
    public float fallPenalty = -1f;
    public float maxEpisodeFallHeight = 0.5f;

    [Header("Stillness Detection")]
    public float stillnessCheckInterval = 1f;
    public float minMovementThreshold = 0.05f;
    public float stillnessPenalty = -0.5f;

    [Header("Control Smoothing")]
    [SerializeField] private float controlSmoothingSpeed = 8f;
    private float smoothedSteer = 0f;
    private float smoothedThrottle = 0f;

    private float previousDistance;
    private float previousSteer;
    private float previousThrottle;

    private Vector3 lastCheckedPosition;
    private float stillnessTimer = 0f;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        startPosition = transform.position;
        startRotation = transform.rotation;
    }

    public override void OnEpisodeBegin()
    {
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        transform.position = startPosition;
        transform.rotation = startRotation;

        controller.ResetSwivel();

        pathHelper.ResetSequence();
        pathHelper.RecalculatePath(transform.position);
        currentWaypointIndex = 0;

        previousDistance = GetCurrentTargetDistance();
        previousSteer = 0f;
        previousThrottle = 0f;
        smoothedSteer = 0f;
        smoothedThrottle = 0f;

        lastCheckedPosition = transform.position;
        stillnessTimer = 0f;
    }

    float GetCurrentTargetDistance()
    {
        if (pathHelper.waypoints == null || pathHelper.waypoints.Length == 0) return 0f;
        Vector3 target = pathHelper.waypoints[Mathf.Min(currentWaypointIndex, pathHelper.waypoints.Length - 1)];
        return Vector3.Distance(transform.position, target);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // --- proprioception: agent's own wheel state ---
        float swivelRad = controller.CurrentSwivelAngleRadians;
        sensor.AddObservation(Mathf.Sin(swivelRad)); // 1
        sensor.AddObservation(Mathf.Cos(swivelRad)); // 1
        sensor.AddObservation(previousSteer);         // 1
        sensor.AddObservation(previousThrottle);      // 1

        if (pathHelper.waypoints == null || pathHelper.waypoints.Length == 0 || pathHelper.AllTargetsVisited)
        {
            sensor.AddObservation(Vector3.zero); // localTargetPosition (3)
            sensor.AddObservation(0f);             // forwardDot (1)
            sensor.AddObservation(0f);             // rightDot (1)
            sensor.AddObservation(0f);             // distance (1)
            sensor.AddObservation(Vector3.zero);  // localVelocity (3)
            sensor.AddObservation(1f);             // curriculum progress (1)
            return;
        }

        Vector3 target = pathHelper.waypoints[Mathf.Min(currentWaypointIndex, pathHelper.waypoints.Length - 1)];
        Vector3 localTargetPosition = transform.InverseTransformPoint(target);
        sensor.AddObservation(localTargetPosition); // 3

        Vector3 targetDirection = (target - transform.position).normalized;
        float forwardDot = Vector3.Dot(transform.forward, targetDirection);
        float rightDot = Vector3.Dot(transform.right, targetDirection);
        sensor.AddObservation(forwardDot); // 1
        sensor.AddObservation(rightDot);   // 1

        sensor.AddObservation(localTargetPosition.magnitude); // 1 (distance)

        Vector3 localVelocity = transform.InverseTransformDirection(rb.linearVelocity);
        sensor.AddObservation(localVelocity); // 3

        sensor.AddObservation((float)pathHelper.currentTargetIndex / Mathf.Max(1, pathHelper.targets.Length)); // 1
    }
    // Total observations = 4 (proprioception) + 3+1+1+1+3+1 (task) = 14 -> Space Size = 14

    public override void OnActionReceived(ActionBuffers actions)
    {
        float targetSteer = actions.ContinuousActions[0];
        float targetThrottle = actions.ContinuousActions[1];

        smoothedSteer = Mathf.MoveTowards(smoothedSteer, targetSteer, controlSmoothingSpeed * Time.fixedDeltaTime);
        smoothedThrottle = Mathf.MoveTowards(smoothedThrottle, targetThrottle, controlSmoothingSpeed * Time.fixedDeltaTime);

        controller.ApplyControl(smoothedSteer, smoothedThrottle);

        if (pathHelper.waypoints == null || pathHelper.waypoints.Length == 0 || pathHelper.AllTargetsVisited)
        {
            previousSteer = smoothedSteer;
            previousThrottle = smoothedThrottle;
            return;
        }

        // --- off-navmesh penalty ---
        NavMeshHit hit;
        bool onMesh = NavMesh.SamplePosition(transform.position, out hit, 0.1f, NavMesh.AllAreas);
        if (!onMesh)
        {
            AddReward(-0.05f);
        }

        // --- stillness penalty (this alone covers "keep moving" pressure) ---
        stillnessTimer += Time.fixedDeltaTime;
        if (stillnessTimer >= stillnessCheckInterval)
        {
            float movedDistance = Vector3.Distance(transform.position, lastCheckedPosition);
            if (movedDistance < minMovementThreshold)
            {
                AddReward(stillnessPenalty);
            }
            lastCheckedPosition = transform.position;
            stillnessTimer = 0f;
        }

        Vector3 target = pathHelper.waypoints[Mathf.Min(currentWaypointIndex, pathHelper.waypoints.Length - 1)];
        float distToWaypoint = Vector3.Distance(transform.position, target);

        Debug.DrawLine(transform.position, target, Color.green);

        // --- exploit-proof progress: the main driver of the reward signal ---
        if (distToWaypoint < previousDistance)
        {
            float progress = previousDistance - distToWaypoint;
            AddReward(progress * 1.5f);
            previousDistance = distToWaypoint;
        }

        // --- single alignment signal: toward the FINAL goal, not the intermediate corner ---
        Vector3 finalGoal = pathHelper.CurrentTarget.position;
        float finalGoalAlignment = Vector3.Dot(transform.forward, (finalGoal - transform.position).normalized);
        AddReward((finalGoalAlignment - 1f) * 0.02f);

        // --- backward-movement penalty ---
        Vector3 velocityDirection = rb.linearVelocity.normalized;
        Vector3 toTargetDirection = (target - transform.position).normalized;
        float velocityAlignment = Vector3.Dot(velocityDirection, toTargetDirection);
        if (rb.linearVelocity.magnitude > 0.05f && velocityAlignment < 0f)
        {
            AddReward(velocityAlignment * 0.05f);
        }

        previousSteer = smoothedSteer;
        previousThrottle = smoothedThrottle;

        if (distToWaypoint < waypointReachDistance)
        {
            currentWaypointIndex++;

            if (currentWaypointIndex >= pathHelper.waypoints.Length)
            {
                AddReward(liftReachBonus);
                pathHelper.AdvanceToNextLift();

                if (pathHelper.AllTargetsVisited)
                {
                    AddReward(allLiftsBonus);
                    EndEpisode();
                }
                else
                {
                    pathHelper.RecalculatePath(transform.position);
                    currentWaypointIndex = 0;
                    previousDistance = GetCurrentTargetDistance();
                }
            }
            else
            {
                AddReward(0.5f);
                previousDistance = GetCurrentTargetDistance();
            }
        }

        if (transform.position.y < startPosition.y - maxEpisodeFallHeight ||
            Vector3.Dot(transform.up, Vector3.up) < 0.3f)
        {
            AddReward(fallPenalty);
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxis("Horizontal");
        continuousActions[1] = Input.GetAxis("Vertical");
    }
}