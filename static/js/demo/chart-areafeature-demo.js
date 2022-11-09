// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';

function number_format(number, decimals, dec_point, thousands_sep) {
  // *     example: number_format(1234.56, 2, ',', ' ');
  // *     return: '1 234,56'
  number = (number + '').replace(',', '').replace(' ', '');
  var n = !isFinite(+number) ? 0 : +number,
    prec = !isFinite(+decimals) ? 0 : Math.abs(decimals),
    sep = (typeof thousands_sep === 'undefined') ? ',' : thousands_sep,
    dec = (typeof dec_point === 'undefined') ? '.' : dec_point,
    s = '',
    toFixedFix = function(n, prec) {
      var k = Math.pow(10, prec);
      return '' + Math.round(n * k) / k;
    };
  // Fix for IE parseFloat(0.55).toFixed(0) = 0;
  s = (prec ? toFixedFix(n, prec) : '' + Math.round(n)).split('.');
  if (s[0].length > 3) {
    s[0] = s[0].replace(/\B(?=(?:\d{3})+(?!\d))/g, sep);
  }
  if ((s[1] || '').length < prec) {
    s[1] = s[1] || '';
    s[1] += new Array(prec - s[1].length + 1).join('0');
  }
  return s.join(dec);
}

// Area Chart Example
var ctx = document.getElementById("myAreaFeature");
var myLineChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: 
          ['v833	', 'v411	', 'v786	', 'v457	', 'v418	', 'v377	', 'v861	', 'v704	', 'v445	', 'v428	', 'v519	', 'v062	', 'v331	', 'v619	', 'v966	', 'v329	', 'v203	', 'v121	', 'v260	', 'v324	', 'v221	', 'v751	', 'v560	', 'v787	', 'v495	', 'v883	', 'v178	', 'v808	', 'v153	', 'v739	', 'v599	', 'v689	', 'v967	', 'v698	', 'v211	', 'v185	', 'v105	', 'v908	', 'v938	', 'v943	', 'v188	', 'v462	', 'v317	', 'v339	', 'v079	', 'v169	', 'v903	', 'v798	', 'v773	', 'v356	', 'v127	', 'v279	', 'v948	', 'v543	', 'v234	', 'v117	', 'v361	', 'v763	', 'v914	', 'v069	', 'v951	', 'v284	', 'v917	', 'v957	', 'v596	', 'v496	', 'v239	', 'v330	', 'v090	', 'v451	', 'v337	', 'v548	', 'v539	', 'v226	', 'v615	', 'v904	', 'v586	', 'v802	', 'v417	', 'v434	', 'v811	', 'v326	', 'v425	', 'v414	', 'v796	', 'v792	', 'v222	', 'v960	', 'v900	', 'v469	', 'v767	', 'v722	', 'v430	', 'v131	', 'v777	', 'v629	', 'v928	', 'v177	', 'v713	', 'v162'],
      datasets: [
          {
              label: "Nicolas",
              lineTension: 0.3,
              backgroundColor: "rgba(255, 165, 0, 0.05)",
              borderColor: "rgba(255, 165, 0, 1)",
              pointRadius: 0.5,
              pointBackgroundColor: "rgba(255, 165, 0, 1)",
              pointBorderColor: "rgba(255, 165, 0, 1)",
              pointHoverRadius: 1,
              pointHoverBackgroundColor: "rgba(255, 165, 0, 1)",
              pointHoverBorderColor: "rgba(255, 165, 0, 1)",
              pointHitRadius: 3,
              pointBorderWidth: 2,
              data:
                  ['0.740477101099861	', '0.720267522115301	', '0.735305572723309	', '0.687868369479601	', '0.713302163067591	', '0.653155684255741	', '0.742003353880504	', '0.696974688493576	', '0.701990318613239	', '0.726631211260239	', '0.683621659969467	', '0.712598512021283	', '0.682690200781697	', '0.681110684888836	', '0.683724447273241	', '0.639026484452208	', '0.649598721441574	', '0.715519461139819	', '0.666487805963435	', '0.657940317240106	', '0.652283819342509	', '0.612307277855495	', '0.616510716028646	', '0.560869203989342	', '0.598832892600483	', '0.548202661117178	', '0.526780188123133	', '0.539207133608933	', '0.535677497884375	', '0.535202262158782	', '0.519594409021920	', '0.544525709403580	', '0.518324265041210	', '0.531419112419786	', '0.570495583415781	', '0.546513213636162	', '0.573205805239226	', '0.594648784254148	', '0.547616153773138	', '0.508593293170873	', '0.499233791012481	', '0.518043691758569	', '0.532009713087725	', '0.546806491455622	', '0.493064510010984	', '0.456619016363119	', '0.487347752859759	', '0.522804527371111	', '0.545218933966309	', '0.585149387141894	', '0.618605871022343	', '0.579052712948097	', '0.572124160461885	', '0.593685763817797	', '0.586562378869326	', '0.589645335479005	', '0.586504872037245	', '0.600953587392579	', '0.547198440268156	', '0.541757248923178	', '0.510573927854458	', '0.493255643246004	', '0.499093889633714	', '0.523553067873946	', '0.512428315628831	', '0.511947317329804	', '0.473649822855830	', '0.447768406226632	', '0.432274529739592	', '0.434043659921993	', '0.414667505583069	', '0.399527103579724	', '0.448732650397595	', '0.462473368233502	', '0.510093493180059	', '0.520892131588827	', '0.493874822385790	', '0.454741085425751	', '0.463903367001394	', '0.461777591783066	', '0.484750446185111	', '0.512219347377831	', '0.500939531312529	', '0.507586918081281	', '0.477386618765959	', '0.524000817930690	', '0.522881400603530	', '0.498217920750547	', '0.471310083925346	', '0.519471916982997	', '0.536466808245766	', '0.553607402410931	', '0.563938853172689	', '0.585441712702121	', '0.584944497019335	', '0.576773060678714	', '0.606968885523976	', '0.585566923987991	', '0.551273782344183	', '0.524309066924286']
          },
    {
      label: "Average",
      lineTension: 0.3,
      backgroundColor: "rgba(78, 115, 223, 0.05)",
      borderColor: "rgba(78, 115, 223, 1)",
      pointRadius: 0.5,
      pointBackgroundColor: "rgba(78, 115, 223, 1)",
      pointBorderColor: "rgba(78, 115, 223, 1)",
      pointHoverRadius: 1,
      pointHoverBackgroundColor: "rgba(78, 115, 223, 1)",
      pointHoverBorderColor: "rgba(78, 115, 223, 1)",
      pointHitRadius: 3,
      pointBorderWidth: 2,
      data: 
          ['0.568755610656606	', '0.557312982754544	', '0.565838494011383	', '0.564061270307265	', '0.591006181208591	', '0.567104482901978	', '0.593600107273339	', '0.563401474417031	', '0.588771588351706	', '0.580460449724725	', '0.550898441981553	', '0.585075572363735	', '0.550787615346753	', '0.584194656150736	', '0.593895081791036	', '0.571590303373797	', '0.581567145760947	', '0.579302037134157	', '0.561370562045608	', '0.589232734011074	', '0.564381814802667	', '0.596218780136809	', '0.564923756902727	', '0.576815159315922	', '0.566682374157579	', '0.561864698107541	', '0.577261961322652	', '0.555384281158876	', '0.556857765776647	', '0.554721373930457	', '0.565544180864761	', '0.590631819288799	', '0.575677884244296	', '0.593323821049642	', '0.599024360334293	', '0.566332632084240	', '0.560980843707893	', '0.575911187045271	', '0.529270373763159	', '0.526545531337150	', '0.523532234284535	', '0.521414308145253	', '0.517640998403823	', '0.514200688044711	', '0.517663498062425	', '0.515634317279183	', '0.516539494930175	', '0.516585133713995	', '0.517882540339211	', '0.513819720555657	', '0.510276899793373	', '0.511421767312643	', '0.507125961760702	', '0.509750399171131	', '0.505472009382811	', '0.509457465211558	', '0.510737199142241	', '0.506005963256805	', '0.508936291890239	', '0.512359590123357	', '0.511765477779440	', '0.509577163306324	', '0.508035471998202	', '0.509248732206398	', '0.505924288892349	', '0.500617154510831	', '0.505035091404318	', '0.509493271865410	', '0.511338777393635	', '0.515098913743846	', '0.513464551942397	', '0.508397923348879	', '0.507201803788177	', '0.508907528317989	', '0.509750691717495	', '0.512515417619117	', '0.513846379851527	', '0.513633494282757	', '0.510018317086995	', '0.507380268822516	', '0.509999477364499	', '0.507858393396211	', '0.502876019678991	', '0.506244261425879	', '0.504973575887400	', '0.501197027033894	', '0.496120288462518	', '0.498033227921188	', '0.496926801388771	', '0.495995113206100	', '0.495809026489016	', '0.496389362322760	', '0.494751572251244	', '0.495059633085849	', '0.490616138635563	', '0.488268883730040	', '0.487477569703879	', '0.491174458350362	', '0.492865945937337	', '0.495591423174879']
     },
 
    {
        label: "Threshold",
        lineTension: 0.3,
        backgroundColor: "rgba(200, 55, 88, 0.05)",
        borderColor: "rgba(200, 55, 88, 1)",
        pointRadius: 0.5,
        pointBackgroundColor: "rgba(200, 55, 88, 1)",
        pointBorderColor: "rgba(200, 55, 88, 1)",
        pointHoverRadius: 1,
        pointHoverBackgroundColor: "rgba(200, 55, 88, 1)",
        pointHoverBorderColor: "rgba(200, 55, 88, 1)",
        pointHitRadius: 3,
        pointBorderWidth: 2,
        data:
            ['0.650000000000000	', '0.650633251611484	', '0.646834122095726	', '0.639568232117809	', '0.640282202699502	', '0.639228745134511	', '0.639642992028773	', '0.638405450374695	', '0.637419467462033	', '0.638373639633953	', '0.636247778932205	', '0.631499732558097	', '0.633448239014890	', '0.627523623281267	', '0.622386149443448	', '0.622717232829502	', '0.622419710303864	', '0.616425501381065	', '0.616073164346779	', '0.614232398273928	', '0.615875072433765	', '0.612911985023187	', '0.609699003251067	', '0.607555971789593	', '0.609351889050924	', '0.604842879937628	', '0.599452907506366	', '0.597352331660024	', '0.596806180096758	', '0.589626019139480	', '0.585375577527514	', '0.578245661970934	', '0.573427176025744	', '0.569959475481518	', '0.563570284533982	', '0.556168966020943	', '0.552617618199074	', '0.547470355874723	', '0.541462138107215	', '0.538560257062449	', '0.530755041392522	', '0.531056067343397	', '0.525059922975097	', '0.521325555237946	', '0.520656440172114	', '0.517924407901433	', '0.512617393014103	', '0.512295745291606	', '0.513839870030018	', '0.507016743742738	', '0.499265625943649	', '0.495486027089201	', '0.488129866693662	', '0.481385893252211	', '0.474310070790617	', '0.468636963525424	', '0.462748894137058	', '0.460847003942068	', '0.456792274469143	', '0.456059258388002	', '0.457672239926499	', '0.457174411969906	', '0.449961809698702	', '0.450641340209882	', '0.446269048848595	', '0.446239120120520	', '0.446102991999537	', '0.441416539475538	', '0.442916454028719	', '0.443201350463604	', '0.441492647933654	', '0.436448884252955	', '0.434618929494795	', '0.430933406317382	', '0.431859909596776	', '0.431654238765028	', '0.430092559421152	', '0.433426153511812	', '0.433361689715720	', '0.434165181674461	', '0.432181940802186	', '0.433178706909621	', '0.433325575602640	', '0.431778862828936	', '0.431592674455363	', '0.433448865438370	', '0.430131569872768	', '0.434699604562696	', '0.433132714924082	', '0.430675507022165	', '0.434692564042469	', '0.432339290352182	', '0.434666022485186	', '0.434866234323965	', '0.431421808136995	', '0.432694043767171	', '0.430062877236917	', '0.430785755384883	', '0.431758233968244	', '0.433865633667326']
          },

     ],
  },
  options: {
    maintainAspectRatio: 1,
    layout: {
      padding: {
        left: 1,
        right: 1,
        top: 1,
        bottom: 1,
      }
    },
    scales: {
      xAxes: [{
        time: {
          unit: 'date'
        },
        gridLines: {
          display: false,
          drawBorder: false
        },
        ticks: {
          maxTicksLimit: 15,
          maxRotation: 0,
          minRotation: 0,
          callback: function(value, index, values) {
            // return '$' + number_format(value);
            return value.substring(0,7);
          }
        }
      }],
      yAxes: [{
        ticks: {
          maxTicksLimit: 100,
          padding: 1,
          // Include a dollar sign in the ticks
          callback: function(value, index, values) {
            // return '$' + number_format(value);
            return value;
          }
        },
        gridLines: {
          color: "rgb(234, 236, 244)",
          zeroLineColor: "rgb(234, 236, 244)",
          drawBorder: false,
          borderDash: [2],
          zeroLineBorderDash: [2]
        }
      }],
    },
    legend: {
      display: true
    },
    tooltips: {
      backgroundColor: "rgb(255,255,255)",
      bodyFontColor: "#858796",
      titleMarginBottom: 10,
      titleFontColor: '#6e707e',
      titleFontSize: 14,
      borderColor: '#dddfeb',
      borderWidth: 1,
      xPadding: 15,
      yPadding: 15,
      displayColors: false,
      intersect: false,
      mode: 'index',
      caretPadding: 10,
      callbacks: {
        label: function(tooltipItem, chart) {
          var datasetLabel = chart.datasets[tooltipItem.datasetIndex].label || '';
          return datasetLabel + ': ' + number_format(tooltipItem.yLabel * 100, 2) + '%';
          // return datasetLabel + ': $' + tooltipItem.yLabel;
        }
      }
    }
  }
});
