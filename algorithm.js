var fs = require("fs");
var path = require("path");
var shell = require('shelljs');

var start = async function(img_path,callback){

    let now_loc = shell.pwd().stdout;
    let img_name = img_path.split('/');
    img_name = img_name[img_name.length-1].split('.')[0];
    
    let command = "bash od.sh " + img_path;
    let od_result = await shell.exec(command,{silent:true}).stdout;
    
    od_result = JSON.parse(od_result);
    console.log(od_result);
    console.log("od over!");
    
    data = {};
    data.main_img = od_result.data[0].dir;
    data.sub_img = [];
    if(od_result.data.length == 1)
    {
        callback(data);
        return ;
    }
    let input_dir = od_result.res_dir;
    let output_dir = path.join(now_loc ,'img_out', 'sub_img_out', img_name);
    command = "bash ss.sh " + input_dir + " " + output_dir;
    
    await shell.exec(command,{silent:true});
    console.log("ss over!");
    
    let levelList = [];
    let files = fs.readdirSync(output_dir);
    let calc_path = "/home/guest/Android-Smoke-app/Android_Module/calc.py "
    for(let i = 0, len = files.length; i < len; i++)
    {
        let url = path.join(output_dir,files[i]);
        command = "python " + calc_path + url;
        levelList[i] = await shell.exec(command,{silent:true}).stdout;
    	  levelList[i] = levelList[i].substring(0,levelList[i].length-1);
    }
     
    for(let i = 1, len = od_result.data.length; i < len; i++)
    {
        data.sub_img.push({url:od_result.data[i].dir,
                           level:levelList[i-1]});
    }
    callback(data);
    return ;
}
module.exports = {start}