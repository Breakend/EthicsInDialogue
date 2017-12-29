// create the tweets

var tweets = [{"timestamp": "2017-11-27T02:02:25", "text": "http://ift.tt/2AbZKWh\u00a0 Ethical Challenges in Data-Driven Dialogue Systems. (arXiv:1711.09050v1 [cs.CL]) #NLProc", "user": "arxiv_cs_cl", "retweets": "1", "replies": "0", "fullname": "cs.CL Papers", "id": "934965602124525568", "likes": "4"}, {"timestamp": "2017-11-27T02:42:03", "text": "Ethical Challenges in Data-Driven Dialogue Systems http://arxiv.org/abs/1711.09050\u00a0", "user": "arxiv_cscl", "retweets": "0", "replies": "0", "fullname": "arXiv CS-CL", "id": "934975580038189057", "likes": "0"}, {"timestamp": "2017-11-27T02:44:35", "text": "Ethical Challenges in Data-Driven Dialogue Systems http://ift.tt/2jqqavB\u00a0", "user": "arXivCS", "retweets": "0", "replies": "0", "fullname": "arXiv cs.*", "id": "934976216733569025", "likes": "0"}, {"timestamp": "2017-11-27T03:00:56", "text": "Ethical Challenges in Data-Driven Dialogue Systems - Peter Henderson http://ift.tt/2jqqavB\u00a0", "user": "deep_rl", "retweets": "0", "replies": "0", "fullname": "Deep RL", "id": "934980328325812224", "likes": "0"}, {"timestamp": "2017-11-27T03:09:26", "text": "Ethical Challenges in Data-Driven Dialogue Systems. Henderson et al. http://arxiv.org/abs/1711.09050\u00a0", "user": "BrundageBot", "retweets": "0", "replies": "1", "fullname": "Brundage Bot", "id": "934982467848634368", "likes": "0"}, {"timestamp": "2017-11-27T03:40:11", "text": "\"Ethical Challenges in Data-Driven Dialogue Systems,\" Henderson et al.: https://arxiv.org/abs/1711.09050\u00a0\n\ncovers \"bias, adversarial examples, privacy, safety, considerations for RL, and reproducibility\"", "user": "Miles_Brundage", "retweets": "3", "replies": "0", "fullname": "Miles Brundage", "id": "934990209657262080", "likes": "13"}, {"timestamp": "2017-11-27T05:41:40", "text": "Ethical Challenges in Data-Driven Dialogue Systems http://arxiv.org/abs/1711.09050\u00a0", "user": "arxiv_cscl", "retweets": "1", "replies": "0", "fullname": "arXiv CS-CL", "id": "935020780215300096", "likes": "0"}, {"timestamp": "2017-11-27T14:32:30", "text": "\"Ethical Challenges in Data-Driven Dialogue Systems\",\nPeter Henderson, Koustuv Sinha, Nicolas Angelard-Gontier, Nan\u2026\nhttps://arxiv.org/abs/1711.09050\u00a0", "user": "arxivml", "retweets": "0", "replies": "0", "fullname": "\u5348\u5f8c\u306earXiv", "id": "935154368789303296", "likes": "0"}, {"timestamp": "2017-11-27T18:41:43", "text": "Ethical Challenges in Data-Driven Dialogue Systems http://arxiv.org/abs/1711.09050\u00a0", "user": "arxiv_cscl", "retweets": "0", "replies": "0", "fullname": "arXiv CS-CL", "id": "935217084300255237", "likes": "0"}, {"timestamp": "2017-11-28T03:11:10", "text": "Excited to show off our new investigative paper from joint MILA/McGill dialogue group: \"Ethical Challenges in Data-Driven Dialogue Systems\". Would love to hear feedback, so feel free to DM me! Big thanks to all co-authors! https://arxiv.org/pdf/1711.09050.pdf\u00a0\u2026", "user": "astro_pyotr", "retweets": "6", "replies": "1", "fullname": "Peter Henderson", "id": "935345295105384449", "likes": "20"}]


  console.log(tweets); // this will show the info it in firebug console
    var div = d3.select(".pub");
    // div.append('br');
    div = div.append('div').classed('hype', true);
    tweets.sum=0;
    tweets.forEach(function (value) {
        tweets.sum += Number(value.likes);//parseInt(value.likes,10);
    }, tweets);
    tweets.retweets=0;
    tweets.forEach(function (value) {
        tweets.retweets += Number(value.retweets);//parseInt(value.likes,10);
    }, tweets);
    var sumLikes = tweets.sum;//'tweets.reduce((x, y) => Number(x.likes) + Number(y.likes));
    div.append('div').classed('heading', true).text('Twitter Hype');
    tweetCount = div.append('div').classed('links', true) ;//.classed('heading', true).style('font-size' , '14px');
    tweetCount.append('i').classed('fa fa-comment mcgill', true).attr('aria-hidden', "true").style('color', '#4099FF');//('<i class="fa fa-heart" aria-hidden="true"></i>');
    tweetCount.append('span').style('font-size' , '14px').text(' ' + tweets.length + ' tweets ');
    // tweetCount.append('br')
    tweetCount.append('i').classed('fa fa-retweet mcgill', true).attr('aria-hidden', "true").style('color', '#17bf63');//('<i class="fa fa-heart" aria-hidden="true"></i>');
    tweetCount.append('span').style('font-size' , '14px').text(' ' + tweets.retweets + ' retweets ');
    // tweetCount.append('br')
    tweetCount.append('i').classed('fa fa-heart mcgill', true).attr('aria-hidden', "true");//('<i class="fa fa-heart" aria-hidden="true"></i>');
    tweetCount.append('span').style('font-size' , '14px').text(' ' + sumLikes + ' likes');
    tweetCount.append('br');
    // tweetCount.append('br');

    // tweetCount.attr('font-color', 'red');

    var tdiv = div.append('div').classed('twdiv', true);
    var tContentContainer = div.append('div').classed('twcontentcontainerdiv', true);
    // var tcontentdiv = tContentContainer.append('div').classed('twcont', true)
    tContentContainer.append('div').classed('heading', true).text('What are they saying?');

     var ix_tweets = tweets; // looks a little weird, i know
     var first = true;
     for(var j=0,m=ix_tweets.length;j<m;j++) {

       var tcontentdiv = tContentContainer.append('div').classed('twcont', true);
       var t = ix_tweets[j];
      //  TODO: add not-boring tweet stuff
       var border_col = true ? '#3c3' : '#fff'; // distinguish non-boring tweets visually making their border green
       var timgdiv = tdiv.append('img').classed('twimg', true).attr('src', "https://twitter.com/" + t.user + "/profile_image?size=normal")
                        //  .attr('style',);
      timgdiv.attr('id', 'tweet'+ t.id);

       elt = tcontentdiv;
       tid = t.id;
       if (first) {
        //  elt.attr('style', 'visibility: hidden;'); // make visible
         elt.attr('style', '  display: inline-block;'); // make visible
         first=false;
       } else {
         elt.attr('style', 'display:none;'); // make visible
       }
       elt.html(''); // clear it
       elt.attr('id', 'tweet' + tid)
       var elem = elt[0][0]
       var twitterdiv = document.createElement('div');
      //  twitterdiv.style.display ;
      elem.append(twitterdiv);
      // elem.style('margin', '0 auto')
       twttr.widgets.createTweet(
         tid, twitterdiv,
         {
           conversation : 'none',    // or all
           cards        : 'hidden',  // or visible
           linkColor    : '#cc0000', // default is blue
           theme        : 'light'    // or dark
         });

        act_fun = function(){
            var elt = this;
            selected =  div.selectAll(".twcont");
            selected.style('visibility','visible');
            selected.style('display','none');
            // console.log(elt.id)
            selectedID = div.selectAll('.twcont#' + elt.id);
            // console.log("Selected" + selected.length);
            // console.log(selected)
            // console.log(selectedID)
          selectedID.attr('style', 'display:inline-block;');
        }
       timgdiv.on('mouseover', act_fun);
       timgdiv.on('click', act_fun);
       timgdiv.on('mouseout', function(elt, col){
         return function() { }
       }(timgdiv, border_col));
     }
