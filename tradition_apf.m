obstalce_radius = 2;
obstacles = [
    4, 3;
    6, 4.6;
    6, 0;
    6, 1;
    6, 2;
    6, 3;
    6, 4;
    6, 5;
    7, 5;
    8, 5;
    9, 5;
    10, 5;
];
scatter(obstacles(:, 1), obstacles(:, 2), 'r', 'filled');
xlim([-1 10]);
ylim([-1 10]);

hold on;
goal = [0 0];
scatter(goal(:, 1), goal(:, 2), 'b', 'filled');

step = 0.1;
tolerance = 0.05;
current_point = [8 6];
while dist(current_point, goal) >= tolerance
    F_att = att_force(current_point, goal);
    
    F_rep = 0;
    for i = 1:length(obstacles)
        F_rep = F_rep + rep_force(current_point, obstacles(i, :));
    end
    
    F = F_att + F_rep;
    F = F / norm(F);
    
    current_point = current_point + F * step;
    scatter(current_point(:, 1), current_point(:, 2), 'g', 'filled');
    drawnow;
    pause(0.1);
end
hold off;

% calc distance
function ret = dist(pt1, pt2)
    ret = sqrt((pt1 - pt2) * (pt1 - pt2)');
end
   
% Attractive Potential
function ret = att_force(current_point, target_point)
    att_gain = 1;
    ret = -att_gain * (current_point - target_point);
end

% Repulsive Potential
function ret = rep_force(current_point, target_point)
    rep_threshold = 1.6;
    d = dist(current_point, target_point);
    
    rep_gain = 1;
    if d < rep_threshold
        ret = rep_gain * (1 / d - 1 / rep_threshold) / power(d, 3) * (current_point - target_point); 
    else
        ret = 0;
    end
end