function str = envsubst(str)
    % Find all environment variable patterns in the format ${VAR_NAME}
    pattern = '\$\{([A-Za-z_][A-Za-z0-9_]*)\}';
    
    % Use regular expression to find matches for environment variables
    tokens = regexp(str, pattern, 'tokens');
    
    % Iterate through all found environment variable placeholders
    for i = 1:length(tokens)
        var_name = tokens{i}{1};  % Extract variable name (without braces)
        var_value = getenv(var_name);  % Get the environment variable value
        var_value = strrep(var_value, '\', '/');
        disp(var_value)
        
        % Replace the placeholder with the environment variable value
        if ~isempty(var_value)
            % Proper replacement syntax for regex match
            str = regexprep(str, ['\$\{' var_name '\}'], var_value);
        else
            error(['Environment variable ' var_name ' is not set.']);
        end
    end
end