int main(){

    //idk I thought about browser last night - particularly how to built firefox
    //essentially - you want a UI framework (Swing, etc.) + bytecode interpreter + SQLLite database
    //UI framework is for the tool_bars + search bars + settings + etc
    //bytecode interpreter is like python - but for javascript - it takes raw code every time we load a site - and interpret it line by line
    //concurrency is multiple tabs - we want to manage the bytecode exec machine internally in the browser - so it's like a mini OS within an OS

    //I thought about building a bytecode interpreter machine - the idea is to serialize statement - so we built something like what I built to compile code - except for this time - we dictionarize literally everything ' from variable name, etc., name of operation
    //so normal compilers compile var_name -> addresses, arithmetic to instructions, etc.
    //bytecode machine has a global dictionary, convert var_name -> path, arthmetic instructions -> enumerated dispatch code
    //so everything is a dictionary
    //like everything is a function - function in bytecode world is actually a string - and its arguments have pre-deterministic paths - says foo(int a, int b) then a has the path _global_foo_a, b has the path _global_foo_b
    //we want to serialize the function - and store the address of the function in the global dictionary - like __global_foo
    //it's a nice project to get a grasp of modern architecture - I'll try to build the bytecode machine someday
    
    //why dictionaries? because they are good for concurrency
    //and it's cheap - for every var access in Python or JavaScript is already 1 CACHE_LINE_ACCESS given their nature of object langauge
}