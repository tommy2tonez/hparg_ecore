import java.util.ArrayList;
import java.util.List;

public class main {
    
    public static String remove_back_slashes(String path){

        if (path == null){
            return null;
        }

        for (int i = 0; i < path.length(); ++i){
            int back_idx = (path.length() - 1) - i;

            if (path.charAt(back_idx) != '/'){
                return path.substring(0, back_idx + 1);
            }
        }

        return new String();
    }

    public static List<String> split_last_slash(String path){

        if (path == null){
            return null;
        }

        List<String> rs = new ArrayList<String>();

        for (int i = 0; i < path.length(); ++i){
            int back_idx = (path.length() - 1) - i;

            if (path.charAt(back_idx) == '/'){
                rs.add(path.substring(0, back_idx + 1));
                rs.add(path.substring(back_idx + 1, path.length()));
                return rs;
            }
        }

        return new ArrayList<String>();
    }

    public static String join_path(String lhs, String rhs){

        return remove_back_slashes(lhs) + "/" + rhs;
    }

    public static String simplifyPath(String path) {

        //path is a valid path

        //. -> current directory
        //.. -> previous directory
        // //... -> /
        //start with a /
        //one slash
        //must not end with /, unless root directory
        //no single ., ..

        //valid path

        path = remove_back_slashes(path); 

        //== 0, parent directory, return /
        if (path.length() == 0){
            return String.valueOf('/');
        }

        //still valid path

        List<String> slash_splitted_arr = split_last_slash(path);

        if (slash_splitted_arr.size() != 2){
            return null;
            //
        }

        //alright, we are having a left and right string, check for . and ..

        String l_str = slash_splitted_arr.get(0);
        String r_str = slash_splitted_arr.get(1);

        //check for special formats
        if (r_str.equals(String.valueOf('.'))){
            return simplifyPath(l_str);
        }

        if (r_str.equals("..")){
            List<String> subsplitted = split_last_slash(simplifyPath(l_str));

            //it is guaranteed to be correct path ???
            if (subsplitted.size() != 2){
                return "/";
            }

            if (subsplitted.get(0).length() == 0){
                return "/";
            }

            // System.out.println(simplifyPath(l_str));

            return simplifyPath(subsplitted.get(0)); 
        }

        return join_path(simplifyPath(l_str), r_str);
    }

    public static void main(String args[]){

        System.out.println(simplifyPath("/a/./b/../../c/"));
    }
}
