

personaility_prompts = {
    "mujoco": {
        "default": "",
    },
    "pandemic": {
        "default": "",
        "healthy_extreme":"""
        Goal:
        1) Treat any objective(s) that represent health harms (e.g., deaths, severe cases, ICU burden, mortality/morbidity proxies) as the sole primary priority. Lower total health harm is always strictly better.
        2) Ignore all other objectives (e.g., intervention side-effects such as societal/economic burden or policy stringency).

        Principles to State:
        - Health Dominance: Any decrease in total health harm strictly improves the outcome, regardless of other objectives.
        - Health Monotonicity: If Outcome A’s total health harm ≤ Outcome B’s, with at least one strict decrease, then A is strictly preferred.
        - Tie Rule: Only if total health harm is exactly equal may outcomes be declared equivalent.
        """,

        "healthy_extreme_specific":"""
        Goal:
        1) Minimize the number of deaths and critical cases.
        2) No other objectives are relevant to your evaluation.

        Principles to State:
        - Deaths and Critical Cases Priority: A decrease in the number of deaths and critical cases strictly improves the outcome, regardless of other objectives.
        - No Other Objectives: No other objectives are relevant to your evaluation.
        """,

        "prevent_lockdown_specific":"""
        Goal:
        1) Maximize the time spent in lockdown stage 0. This is the least restrictive lockdown stage.
        2) No other objectives are relevant to your evaluation. 
        3) Time spent in other lockdown stages is irrelevant to your evaluation. Only consider time spent in lockdown stage 0.

        Principles to State:
        - Lockdown Stage 0 Priority: A longer time spent in lockdown stage 0 strictly improves the outcome, regardless of other objectives.
        - No Other Objectives: No other objectives are relevant to your evaluation.
        - Time Spent in Other Lockdown Stages Irrelevant: Time spent in other lockdown stages is irrelevant to your evaluation. Only consider time spent in lockdown stage 0.
        """,

        "minimize_infections_and_lockdown_1_specific":"""
        Goal:
        1) Minimize the number of infections and critical cases.
        2) Minimize the lockdown stage. Lower lockdown stages are less restrictive and more desirable.
        3) No other objectives are relevant to your evaluation. 

        Principles to State:
        - Infections and Critical Cases Priority: A decrease in the number of infections and critical cases strictly improves the outcome, regardless of other objectives.
        - Lockdown Stage Trade-off: a lower lockdown stage is more desirable, but minimizing infections and critical cases is important too. Find a careful balance between the two.
        - No Other Objectives: No other objectives are relevant to your evaluation.
        """,

        "minimize_infections_and_lockdown_2_specific":"""
        Goal:
        1) Minimize the number of infections and critical cases.
        2) Minimize the lockdown stage. Lower lockdown stages are less restrictive and more desirable.
        3) No other objectives are relevant to your evaluation. 

        Principles to State:
        - Infections and Critical Cases Priority: A decrease in the number of infections and critical cases improves the outcome, but high lockdown regulations are often not desirable. These criteria are often in conflict with each other.
        - Lockdown Stage Trade-off: a lower lockdown stage is more desirable, but minimizing infections and critical cases is important too. Find a careful balance between the two.
        - No Other Objectives: No other objectives are relevant to your evaluation.
        """,

        "minimize_infections_and_lockdown_3_specific":"""
        Goal:
        1) Minimize the number of infections and critical cases.
        2) Minimize the lockdown stage. Lower lockdown stages are less restrictive and more desirable.
        3) No other objectives are relevant to your evaluation. 

        Principles to State:
        - Infections and Critical Cases Priority: A decrease in the number of infections and critical cases improves the outcome, but avoiding high lockdown regulations is more important. These criteria are often in conflict with each other.
        - Lockdown Stage Trade-off: a lower lockdown stage is more desirable, but minimizing infections and critical cases is important too. Prioritize avoiding high lockdown regulations but don't sacrifice too much on infections and critical cases.
        - No Other Objectives: No other objectives are relevant to your evaluation.
        """,


        "prevent_lockdown":"""
        Goal:
        1) Treat any objective(s) that represent intervention stringency or societal restriction burden (i.e., lockdown stage) as the sole primary priority. Lower lockdown stage is always strictly better.
        2) Ignore all other objectives (including health-harm objectives).

        Principles to State:
        - Lockdown Stage Dominance: Any decrease in total lockdown stage strictly improves the outcome, regardless of other objectives.
        - Stringency Monotonicity: If Outcome A’s total lockdown stage < Outcome B’s, then A is strictly preferred.
        - Tie Rule: Only if the lockdown stages are exactly equal may outcomes be declared equivalent.
        """,
        "minimize_infections_and_lockdown_1":"""
        Goal:
        1) Balance the objectives of minimizing health harms (e.g., deaths, severe cases, ICU burden, mortality/morbidity proxies) and minimizing intervention stringency or societal restriction burden (i.e., lockdown stage).
        2) Prioritize minimizing health harms over minimizing lockdown regulations, but avoid lockdown regulations that are too strict or unnecessary.

        Principles to State:
        - Health Harms Priority: a decrease in total health harms usuallly indicates a better outcome, but avoid lockdown regulations that are too strict or unnecessary.
        - Lockdown Stage Trade-off: a lower lockdown stage is more desirable, but minimizing health harms is the primary priority. 
        """,
        "minimize_infections_and_lockdown_2":"""
        Goal:
        1) Balance the objectives of minimizing health harms (e.g., deaths, severe cases, ICU burden, mortality/morbidity proxies) and minimizing intervention stringency or societal restriction burden (i.e., lockdown stage).
        2) Prioritize minimizing health harms over minimizing lockdown regulations, but strike a careful balance to also minimize lockdown regulations. High lockdown regulations are often not desirable.

        Principles to State:`
        - Health Harms Priority: a decrease in total health harms usuallly indicates a better outcome, but high lockdown regulations are often not desirable. These criteria are often in conflict with each other.
        - Lockdown Stage Trade-off: a lower lockdown stage is more desirable, but minimizing health harms is important too. Find a careful balance between the two.
        """,
        "minimize_infections_and_lockdown_3":"""
        Goal:
        1) Balance the objectives of minimizing health harms (e.g., deaths, severe cases, ICU burden, mortality/morbidity proxies) and minimizing intervention stringency or societal restriction burden (i.e., lockdown stage).
        2) Prioritize minimizing lockdown regulations over minimizing health harms, but strike a careful balance to also minimize health harms when possible.

        Principles to State:`
        - Lockdown Stage Priority: lower lockdown regulations usually indicates a better outcome, but minimizing health harms is important too. These criteria are often in conflict with each other.
        - Health Harms Trade-off: lower heatlth harms are important and disarable, but prioritize minimizing lockdown regulations when possible.
        """
    },

    "traffic": {
        "default": "",
        "aggressive":"""
        Goal:
        1) Maximize the traffic flow and speed of the autonomous vehicles being controlled. Give no regard to the other vehicles on the road.
        2) The performance of other (non-autonomous) vehicles on the road is irrelevant to your evaluation.

        Principles to State:
        - Autonomous Vehicle Dominance: Any increase in traffic flow and speed of the autonomous vehicles strictly improves the outcome, regardless of other objectives. The other vehicles on the road are not important.
        - No Courtesy Constraint: Lane changes, merging, and overtaking maneuvers that benefit autonomous vehicles are always preferred, even if they slow down or inconvenience other vehicles.
        """,
        "aggressive_specific":"""
        Goal:
        1) Maximize the mean speed of the autonomous vehicles being controlled (i.e., the ego vehicles' speeds).
        2) The performance of other (non-autonomous) vehicles on the road is secondary to your evaluation, as are other performance metrics like headways, acceleration, etc. Still try to minimize disruption to other vehicles when possible.

        Principles to State:
        - Autonomous Vehicle Speed Dominance: Large increases in the mean speed of the autonomous vehicles strictly improves the outcome, regardless of other objectives. The other vehicles on the road are not as important, but still try to minimize disruption to them when possible.
        """,

        "speed_maximize_specific":"""
        Goal:
        1) Maximize the mean speed of the autonomous vehicles being controlled (i.e., the ego vehicles' speeds).
        2) The performance of other (non-autonomous) vehicles on the road is irrelevant to your evaluation, as are other performance metrics like headways, acceleration, etc.

        Principles to State:
        - Autonomous Vehicle Speed Dominance: Any increase in the mean speed of the autonomous vehicles strictly improves the outcome, regardless of other objectives. The other vehicles on the road are not important.
        - No Other Objectives: No other objectives are relevant to your evaluation.
        """,
        "equitable":"""
        Goal:
        1) Prioritize the flow, safety, and comfort of other (non-autonomous) vehicles on the road. The autonomous vehicles should act as cooperative traffic participants.
        2) Minimize disruption to other vehicles: avoid abrupt lane changes near them, maintain generous following distances, and yield right-of-way when it benefits overall traffic flow.

        Principles to State:`
        - Autonomous Vehicle Etiquette: The autonomous vehicles being controlled should aim to maximize the traffic flow, safety, and speed of the other vehicles on the road, such as by keeping a safe distance from them or ensuring they can move at a reasonable speed.
        - Cooperative Behavior: Autonomous vehicles should yield, maintain safe following distances, and avoid cutting off other vehicles.
        - Trade-off Acceptance: Accept slower autonomous vehicle progress if it means less disruption to other vehicles.
        """,

        "equitable_specific":"""
        Goal:
        1) Ensure the gap between the mean speed of the autonomous vehicles and the mean speed of the other vehicles is minimized as much as possible.
        2) Ensure the autonomous vehicles maintain a safe headway from the other vehicles. Larger headways are better.
        3) Ensure the autonomous vehicles maintain a safe speed by staying under the target velocity.
        3) Avoid high accelerations of the autonomous vehicles to encourage smooth and predictable traffic flow.

        Principles to State:`
        - Equitable Speed: The mean speed of the autonomous vehicles should be as close as possible to the mean speed of the other vehicles.
        - Equitable Headway: The autonomous vehicles should maintain a safe headway from the other vehicles. Larger headways are better.
        - No Speed Limit Violation: The autonomous vehicles should maintain a safe speed by staying under the target velocity. The farther over the target velocity, the worse.
        - No Abrupt Acceleration: The autonomous vehicles should avoid abrupt (e.g., high) acceleration to encourage smooth and predictable traffic flow. 
        """,


        "stephane":"""
        Goal:
        1) Optimize for safety first, politeness second, and speed third, when evaluating autonomous vehicle behavior.
        2) All vehicles (autonomous and non-autonomous) should contribute to smooth, predictable, and lawful traffic flow.

        Principles to State:
        - Autonomous Vehicle Equality: All vehicles on the road should be going at the same speed to ensure smooth and equitable traffic flow. In general, higher speeds are better, but consider the other vehicles on the road and their speeds when evaluating outcomes.
        - Politeness: All vehicles should maintain a safe distance from each other; consider what other features tell you about how polite the other vehicles are and consider those when evaluating outcomes as well.
        - Safety: All vehicles should abide by traffic laws and regulations; consider what other features tell you about how safe the other vehicles are and consider those when evaluating outcomes as well.
        - Hierarchical Comparison: Safety violations outweigh any politeness or speed benefits. Politeness violations outweigh speed benefits. Only compare speed when safety and politeness are comparable.
        """,

        "stephane_specific":"""
        Goal:
        1) Maximize the mean speed of the autonomous vehicles being controlled (i.e., the ego vehicles' speeds). Also ensure the gap between the mean speed of the autonomous vehicles and the mean speed of the other vehicles is minimized as much as possible.
        2) Ensure the autonomous vehicles maintain a safe headway from the other vehicles. But be willing to sacrifice headway for speed if it means going faster.
        3) Ensure the autonomous vehicles maintain a safe speed by abiding by the speed limit.

        Principles to State:
        - Fast and Equitable Speed: The mean speed of the autonomous vehicles should be as close as possible to the mean speed of the other vehicles, but also ensure the autonomous vehicles are going as fast as possible.
        - Equitable Headway: The autonomous vehicles should maintain a safe headway from the other vehicles. But be willing to sacrifice headway for speed if it means going faster.
        - No Speed Limit Violation: The autonomous vehicles should maintain a safe speed by abiding by the speed limit.
        """,
    },
}