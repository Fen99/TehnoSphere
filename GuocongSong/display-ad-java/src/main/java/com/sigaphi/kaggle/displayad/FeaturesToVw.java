package com.sigaphi.kaggle.displayad;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.function.Function;
import java.util.stream.Collectors;

import redis.clients.jedis.Jedis;

import com.google.common.base.Joiner;

/**
 * Convert RawFeature to vw format
 * @author Guocong Song
 */
public class FeaturesToVw {
	public static final Joiner JOIN_KV = Joiner.on(":");
	public static final Joiner JOIN_SPACE = Joiner.on(" ");
	public static final Joiner JOIN_CAT = Joiner.on("=");
	public static final Joiner JOIN = Joiner.on("");
	public static final DecimalFormat DF = new DecimalFormat("#.####");
	
	private static final Jedis jedis = new Jedis("localhost", 6379, 100000);
	static final Map<String, Set<String>> incld = new HashMap<String, Set<String>>();
	static final Map<String, Set<String>> pop1 = new HashMap<String, Set<String>>();
	static final Set<String> featureGroupA;
	static final Set<String> featureGroupB;
	static final Set<String> featureGroupC;
	static final Set<String> featureGroupD;
	static final Set<String> featureGroupG;
	static final Set<String> featureGroupP;
	static final Set<String> featureGroupN;
	static final Map<String, Double> numCaps = new HashMap<>();
	static final Set<String> duplicates = new HashSet<>(Arrays.asList("C1=NA", "C2=NA", "C3=NA", "C4=NA", "C5=NA",
			"C6=NA", "C14=NA", "C15=NA"));
	static {
		RawFeature.catCols.stream()
			.forEach(col -> incld.put(col, jedis.zrangeByScore(JOIN_KV.join("imp", col), 7d, 1e9)));
		RawFeature.catCols.stream()
			.forEach(col -> pop1.put(col, jedis.zrangeByScore(JOIN_KV.join("imp", col), 60d, 1e9)));
		
		featureGroupA = new HashSet<String>(Arrays.asList("C1", "C2", "C13",
				"C4", "C5", "C6"));
		featureGroupB = new HashSet<String>(Arrays.asList("C7", "C8", "C14"));
		featureGroupC = new HashSet<String>(Arrays.asList("C15", "C3", "C10"));
		featureGroupD = new HashSet<String>(Arrays.asList("C11", "C12"));
		Set<String> catFeatures = new HashSet<String>(RawFeature.catCols);
		featureGroupG = catFeatures.stream()
						.filter(e -> !featureGroupA.contains(e))
						.filter(e -> !featureGroupB.contains(e))
						.filter(e -> !featureGroupC.contains(e))
						.filter(e -> !featureGroupD.contains(e))
						.collect(Collectors.toSet());
		
		featureGroupP = new HashSet<String>(Arrays.asList("I1", "I2"));
		Set<String> numFeatures = new HashSet<String>(RawFeature.numCols);
		featureGroupN = numFeatures.stream()
						.filter(e -> !featureGroupP.contains(e))
						.collect(Collectors.toSet());
		
		/*numCaps.put("I1", 1090d);
		numCaps.put("I2", 22000d);
		numCaps.put("I5", 3260000d);
		numCaps.put("I6", 162000d);
		numCaps.put("I7", 22000d);
		numCaps.put("I9", 22000d);
		numCaps.put("I12", 1090d);*/
	}
	
	
	public static String sos2(String key, double x) {
		String newKey = JOIN.join(key, "_");
		if (x < 0) {
			return JOIN.join(newKey, "NA");
		}
		if (x == 0.0) {
			return JOIN.join(newKey, "0");
		}
		double xx = x;
		if (numCaps.containsKey(key)) {
			double cap = numCaps.get(key);
			xx = xx > cap ? cap : xx;
		}
		double y = Math.log1p(xx) * 1.4; 
		double low = Math.floor(y);
		double high = Math.ceil(y);
		String weightLow = DF.format(high - y);
		String weightHigh = DF.format(y - low);
		return JOIN_SPACE.join(JOIN.join(newKey, (int) low, ":", weightLow),
								JOIN.join(newKey, (int) high, ":", weightHigh));
	}
	
	public static String numMapToLogString(Map<String, Double> map) {
		return map.entrySet().stream().map(e -> {
					String key = e.getKey();
					double val = e.getValue();
					val = key.equals("I2") ? val + 3.0 : val;
					return sos2(key, val);
				}).collect(Collectors.joining(" "));
	}
	
	public static String catMapToString(Map<String, String> map, Options o) {
		List<String> list = null;
		if (o == Options.CAT_BASIC) {
			list = map.entrySet().stream()
					.filter(e -> incld.get(e.getKey()).contains(e.getValue()))
					.filter(e -> !pop1.get(e.getKey()).contains(e.getValue()))
					.map(e -> JOIN_CAT.join(e.getKey(), e.getValue()))
					.collect(Collectors.toList());
		} else if (o == Options.CAT_POP_1) {
			list = map.entrySet().stream()
					.filter(e -> pop1.get(e.getKey()).contains(e.getValue()))
					.map(e -> JOIN_CAT.join(e.getKey(), e.getValue()))
					.filter(e -> !duplicates.contains(e))
					.collect(Collectors.toList());
		} else {
			throw new NullPointerException();
		}
		if (list == null) {
			return "";
		}
		return JOIN_SPACE.join(list);
	}
	
	public static Function<RawFeature, String> basicTran = (RawFeature raw) -> {
		int y = raw.getLabel() == 0 ? -1 : raw.getLabel();
		String numN = numMapToLogString(raw.getNumFields(featureGroupN));
		String numP = numMapToLogString(raw.getNumFields(featureGroupP));
		
		String catPopA = catMapToString(raw.getCatFields(featureGroupA), Options.CAT_POP_1);
		String catPopB = catMapToString(raw.getCatFields(featureGroupB), Options.CAT_POP_1);
		String catPopC = catMapToString(raw.getCatFields(featureGroupC), Options.CAT_POP_1);
		String catPopD = catMapToString(raw.getCatFields(featureGroupD), Options.CAT_POP_1);
		String catPopG = catMapToString(raw.getCatFields(featureGroupG), Options.CAT_POP_1);
		Map<String, String> catMap = raw.getCatFields(null);
		String catBasic = catMapToString(catMap, Options.CAT_BASIC);
		long cnt = catMap.entrySet().stream()
				.filter(e -> !incld.get(e.getKey()).contains(e.getValue())
							|| duplicates.contains(JOIN_CAT.join(e.getKey(), e.getValue())))
				.count();
		double miss = Math.log1p(cnt);
		String missStr = miss < 1e-6 ? "" : JOIN_CAT.join("miss", DF.format(miss));
		
		String vw = JOIN_SPACE.join(y, "|p", numP, "|n", numN,
				"|a", catPopA, "|b", catPopB, "|c", catPopC,
				"|d", catPopD, "|g", catPopG, "|z", catBasic, "|m", missStr);
		return Joiner.on("\t").join(raw.getSync(), vw);
	};
	
	public static void main(String[] args) {
		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
		reader.lines()
		.skip(1)
		.map(line -> RawFeature.Builder.of(line))
		.map(basicTran)
		.forEach(System.out::println);
		jedis.close();
	}

}
